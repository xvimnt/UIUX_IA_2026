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
import uuid
import wave
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PORT", 8765))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
AUDIO_CACHE_DIR = os.path.join(STATIC_DIR, "audio_cache")
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

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


class AvatarRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def do_POST(self):
        if self.path == "/api/speak":
            content_len = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_len) if content_len > 0 else b"{}"
            try:
                payload = json.loads(body_bytes.decode("utf-8"))
            except Exception:
                payload = {}

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
