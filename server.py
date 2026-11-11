#!/usr/bin/env python3
"""
Minimalist, zero-dependency Backend API & Static Server for Girl12 3D Lip-Sync Avatar.
Run with:
    python3 server.py

Endpoints:
- GET  /           -> Serves the 3D Web Frontend (static/index.html)
- POST /api/speak  -> Accepts {"text": "..."}
                      Returns {"text": "...", "duration": float, "audio_url": str|null, "visemes": [...]}
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import urllib.request
import urllib.error
import uuid
import wave
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_env_file():
    env_path = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip()
                    if k not in os.environ:
                        os.environ[k] = v

_load_env_file()

PORT = int(os.environ.get("PORT", 8765))
STATIC_DIR = os.path.join(BASE_DIR, "static")
AUDIO_CACHE_DIR = os.path.join(STATIC_DIR, "audio_cache")
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
# Fastest ultra-low-latency model on OpenRouter (benchmarked ~0.2s response time)
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.2-1b-instruct")
# Multimodal audio-in model on OpenRouter for direct voice microphone input
OPENROUTER_AUDIO_MODEL = os.environ.get("OPENROUTER_AUDIO_MODEL", "google/gemini-2.5-flash")

# CC3+ Viseme mapping (matches Girl12's morph targets in girl12.glb)
CHAR_TO_VISEME = {
    "a": {"V_Open": 0.75, "V_Wide": 0.25},
    "e": {"V_Wide": 0.65, "V_Open": 0.30},
    "i": {"V_Wide": 0.80, "V_Open": 0.15},
    "o": {"V_Tight_O": 0.80, "V_Open": 0.35},
    "u": {"V_Tight_O": 0.85, "V_Tight": 0.30},
    "y": {"V_Wide": 0.60, "V_Open": 0.20},
    "b": {"V_Explosive": 0.90},
    "p": {"V_Explosive": 0.95},
    "m": {"V_Explosive": 0.85},
    "f": {"V_Dental_Lip": 0.85},
    "v": {"V_Dental_Lip": 0.80},
    "t": {"V_Dental_Lip": 0.35, "V_Wide": 0.30, "V_Lip_Open": 0.25},
    "d": {"V_Dental_Lip": 0.35, "V_Open": 0.25},
    "n": {"V_Dental_Lip": 0.30, "V_Lip_Open": 0.30},
    "l": {"V_Lip_Open": 0.55, "V_Open": 0.25},
    "s": {"V_Tight": 0.45, "V_Wide": 0.35},
    "z": {"V_Tight": 0.45, "V_Wide": 0.35},
    "c": {"V_Affricate": 0.65, "V_Wide": 0.20},
    "j": {"V_Affricate": 0.75},
    "g": {"V_Open": 0.35, "V_Lip_Open": 0.40},
    "k": {"V_Open": 0.35, "V_Lip_Open": 0.40},
    "q": {"V_Tight_O": 0.70, "V_Open": 0.20},
    "r": {"V_Tight": 0.55, "V_Lip_Open": 0.30},
    "w": {"V_Tight_O": 0.80},
    "h": {"V_Open": 0.35, "V_Lip_Open": 0.30},
}

DIGRAPH_TO_VISEME = {
    "th": {"V_Dental_Lip": 0.65, "V_Lip_Open": 0.35},
    "sh": {"V_Affricate": 0.85, "V_Lip_Open": 0.25},
    "ch": {"V_Affricate": 0.85, "V_Lip_Open": 0.25},
    "oo": {"V_Tight_O": 0.90, "V_Open": 0.25},
    "ee": {"V_Wide": 0.85, "V_Open": 0.25},
    "ea": {"V_Wide": 0.85, "V_Open": 0.25},
}


def text_to_viseme_timeline(text: str, target_duration: float | None = None) -> tuple[list[dict], float]:
    """
    Convert arbitrary text into a timestamped sequence of CC3+ viseme weights.
    If `target_duration` is provided (from the voice model's audio clip), scales the timeline
    to match the exact audio length.
    """
    raw_events = []
    cur_t = 0.0
    base_char_dur = 0.075  # ~75ms per character baseline

    i = 0
    text_lower = text.lower()
    while i < len(text_lower):
        ch = text_lower[i]
        digraph = text_lower[i : i + 2]

        if digraph in DIGRAPH_TO_VISEME:
            weights = DIGRAPH_TO_VISEME[digraph]
            dur = base_char_dur * 1.5
            step = 2
        elif ch in CHAR_TO_VISEME:
            weights = CHAR_TO_VISEME[ch]
            dur = base_char_dur * (1.35 if ch in "aeiou" else 0.9)
            step = 1
        elif ch in (" ", ",", ".", "!", "?", ";", ":"):
            weights = {}
            dur = base_char_dur * (1.4 if ch == " " else 3.2)
            step = 1
        else:
            i += 1
            continue

        raw_events.append({
            "start": cur_t,
            "end": cur_t + dur,
            "weights": weights,
        })
        cur_t += dur
        i += step

    total_raw = max(cur_t, 0.1)
    scale = (target_duration / total_raw) if (target_duration and target_duration > 0.1) else 1.0

    scaled_events = [
        {
            "start": round(ev["start"] * scale, 4),
            "end": round(ev["end"] * scale, 4),
            "weights": ev["weights"],
        }
        for ev in raw_events
    ]
    final_duration = round(total_raw * scale, 4)
    return scaled_events, final_duration


def synthesize_voice_model_placeholder(text: str) -> tuple[str | None, float | None]:
    """
    ===========================================================================
    VOICE MODEL INTEGRATION POINT FOR YOUR COWORKER
    ===========================================================================
    Replace this function with your coworker's TTS / Voice Model inference call!
    It should save a `.wav` file into `AUDIO_CACHE_DIR` and return:
        (relative_url_path, duration_in_seconds)

    Out of the box on macOS, if `say` is available, we generate a real `.wav`
    file so you can test both real audio waveform + phoneme lip-sync right now.
    """
    filename = f"speech_{uuid.uuid4().hex[:8]}.wav"
    filepath = os.path.join(AUDIO_CACHE_DIR, filename)

    try:
        # macOS built-in neural/system voice fallback for immediate local testing
        res = subprocess.run(
            ["say", "-o", filepath, "--data-format=LEI16@22050", text],
            check=False,
            capture_output=True,
            timeout=10,
        )
        if res.returncode == 0 and os.path.exists(filepath):
            with wave.open(filepath, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)
            return f"/audio_cache/{filename}", duration
    except Exception:
        pass

    return None, None


def query_openrouter_llm(user_message: str, history: list[dict] | None = None) -> str:
    """
    Send prompt to OpenRouter API using the fastest LLM (Llama 3.2 1B Instruct, ~200ms TTFT)
    to generate natural, conversational responses for the 3D avatar.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are Girl12, an expressive and friendly 3D AI companion. "
                "Keep your answers concise, engaging, and spoken-style (1 to 3 short sentences max) "
                "so that your speech and lip-sync animation flow naturally without long pauses."
            ),
        }
    ]
    if history:
        for msg in history[-6:]:
            if msg.get("role") in ("user", "assistant") and msg.get("content"):
                messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "max_tokens": 120,
        "temperature": 0.7,
    }

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8765",
            "X-Title": "Girl12 3D Avatar",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"OpenRouter API error: {e}")
        return f"I heard you say: {user_message}. (OpenRouter fallback)"


def query_openrouter_audio(audio_base64: str, audio_format: str = "wav", history: list[dict] | None = None) -> str:
    """
    Send raw recorded microphone audio directly to OpenRouter multimodal audio model
    (google/gemini-2.5-flash) which understands speech directly without browser speech recognition dependencies!
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are Girl12, an expressive and friendly 3D AI companion. "
                "Listen to the user speaking in the audio clip and respond concisely in 1 to 2 spoken sentences "
                "so your speech flows naturally."
            ),
        }
    ]
    if history:
        for msg in history[-4:]:
            if msg.get("role") in ("user", "assistant") and msg.get("content"):
                messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": "Listen to my voice in this audio and respond to what I said:"},
            {
                "type": "input_audio",
                "input_audio": {
                    "data": audio_base64,
                    "format": audio_format,
                },
            },
        ],
    })

    payload = {
        "model": OPENROUTER_AUDIO_MODEL,
        "messages": messages,
        "max_tokens": 120,
        "temperature": 0.7,
    }

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8765",
            "X-Title": "Girl12 3D Avatar",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=16) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"OpenRouter Audio API error: {e}")
        return "I could hear your audio, but I had trouble understanding the words. Could you say that again?"


class AvatarRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_len) if content_len > 0 else b"{}"
        try:
            payload = json.loads(body_bytes.decode("utf-8"))
        except Exception:
            payload = {}

        if self.path == "/api/audio-chat":
            audio_base64 = (payload.get("audio_base64") or "").strip()
            audio_format = payload.get("audio_format") or "wav"
            history = payload.get("history") or []
            use_server_tts = bool(payload.get("use_server_tts", True))

            if not audio_base64:
                self._send_json(400, {"error": "Please provide 'audio_base64'."})
                return

            # 1. Listen and respond directly using OpenRouter Audio Multimodal Model
            llm_reply = query_openrouter_audio(audio_base64, audio_format=audio_format, history=history)

            # 2. Synthesize avatar speaking voice
            audio_url, audio_dur = (None, None)
            if use_server_tts:
                audio_url, audio_dur = synthesize_voice_model_placeholder(llm_reply)

            # 3. Compute viseme lip-sync timeline scaled to exact audio length
            visemes, duration = text_to_viseme_timeline(llm_reply, target_duration=audio_dur)

            self._send_json(
                200,
                {
                    "reply_text": llm_reply,
                    "duration": duration,
                    "audio_url": audio_url,
                    "visemes": visemes,
                    "model": OPENROUTER_AUDIO_MODEL,
                },
            )
        elif self.path == "/api/chat":
            user_message = (payload.get("message") or payload.get("text") or "").strip()
            history = payload.get("history") or []
            use_server_tts = bool(payload.get("use_server_tts", True))

            if not user_message:
                self._send_json(400, {"error": "Please provide non-empty 'message'."})
                return

            # 1. Generate answer via OpenRouter LLM
            llm_reply = query_openrouter_llm(user_message, history=history)

            # 2. Synthesize voice audio (via voice model or system TTS)
            audio_url, audio_dur = (None, None)
            if use_server_tts:
                audio_url, audio_dur = synthesize_voice_model_placeholder(llm_reply)

            # 3. Compute viseme lip-sync timeline scaled to exact audio length
            visemes, duration = text_to_viseme_timeline(llm_reply, target_duration=audio_dur)

            self._send_json(
                200,
                {
                    "user_message": user_message,
                    "reply_text": llm_reply,
                    "duration": duration,
                    "audio_url": audio_url,
                    "visemes": visemes,
                    "model": OPENROUTER_MODEL,
                },
            )
        elif self.path == "/api/speak":
            text = (payload.get("text") or "").strip()
            use_server_tts = bool(payload.get("use_server_tts", True))

            if not text:
                self._send_json(400, {"error": "Please provide non-empty 'text'."})
                return

            audio_url, audio_dur = (None, None)
            if use_server_tts:
                audio_url, audio_dur = synthesize_voice_model_placeholder(text)

            visemes, duration = text_to_viseme_timeline(text, target_duration=audio_dur)

            self._send_json(
                200,
                {
                    "text": text,
                    "duration": duration,
                    "audio_url": audio_url,
                    "visemes": visemes,
                },
            )
        else:
            self._send_json(404, {"error": "Not found"})

    def _send_json(self, status_code: int, data: dict):
        encoded = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(encoded)


if __name__ == "__main__":
    print(f"Starting Girl12 3D Lip-Sync Server on http://localhost:{PORT}")
    server = ThreadingHTTPServer(("0.0.0.0", PORT), AvatarRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
