# UIUX_IA_2026 — 3D Lip-Sync & Voice Avatar (Blender + Three.js + Python API)

An interactive 3D conversational avatar built around the **Girl12 (`RL_CC3_Plus`)** character rig.
It synchronizes **56 facial morph targets (`V_*` visemes, blinks, smiles, brows)**, **skeletal jaw/teeth rotation (`CC_Base_JawRoot`)**, and **conversational hand gestures (`Body_Stand_And_Talk`)** with text or voice audio in real time.

---

## 1. How to Run the Project Locally

This project has **zero external dependencies** (uses Python's standard library on the backend and Three.js via ES Module CDN in the browser).

```bash
# 1. Clone the repository
git clone https://github.com/xvimnt/UIUX_IA_2026.git
cd UIUX_IA_2026

# 2. Start the server (defaults to port 8765, or set PORT=8000)
python3 server.py
```

Open **http://localhost:8765** in your browser:
- **Type any text** and click **Speak Text** to see her read the text, move her lips and jaw in sync, and gesture naturally with her hands while standing still.
- **Click `🎙️ Hold / Click to Talk`** to speak into your microphone and have her respond and lip-sync automatically.
- Use the camera preset buttons (**Upper Body + Hands**, **Face Close-Up**, **Full Standing**) to inspect the facial visemes and hand gestures.

---

## 2. Project Architecture

```text
UIUX_IA_2026/
├── server.py               # Zero-dependency Python HTTP & Lip-Sync API server (POST /api/speak)
├── static/
│   ├── index.html          # Three.js real-time 3D frontend + Web Speech / Mic integration
│   └── girl12.glb          # Web-optimized 3D character (10.5 MB, 56 shape keys + Body_Stand_And_Talk animation)
├── lipsync_engine.py       # Blender Python script for procedural lip-sync & hand gestures inside Blender
└── README.md
```

### How the 3D Lip-Sync & Hand Gestures Work
1. **Standing + Hand Gestures (`Body_Stand_And_Talk`)**:
   - Exported from Blender into `static/girl12.glb`.
   - Her lower body (`CC_Base_Hip`, `Pelvis`, `Thigh`, `Calf`, `Foot`) is locked in a natural standing pose.
   - When `isSpeaking` is `false`, the animation mixer smoothly interpolates to frame `0` (hands resting calmly at her sides).
   - When `isSpeaking` is `true`, the mixer plays frames `12–600`, raising her forearms and hands in front of her torso with natural wrist/finger emphasis gestures.
2. **Mouth Visemes (`CC_Base_Body`) + Jaw Bone (`CC_Base_JawRoot`)**:
   - In Character Creator 3+ (`RL_CC3_Plus`) rigs, the 8 viseme morph targets (`V_Open`, `V_Explosive`, `V_Dental_Lip`, `V_Tight_O`, `V_Tight`, `V_Wide`, `V_Affricate`, `V_Lip_Open`) shape the upper and lower lips, while the lower teeth (`CC_Base_Teeth02`) and chin are weighted to the `CC_Base_JawRoot` bone.
   - Both `lipsync_engine.py` (in Blender) and `static/index.html` (in Three.js) automatically drive `CC_Base_JawRoot` rotation in sync with the active viseme weights so her teeth and jaw open naturally on vowels.

---

## 3. How to Incorporate the Voice Model & Talk Using the Microphone

There are two complementary ways to connect your coworker's **Voice Model (STT + LLM + TTS or Speech-to-Speech)** to this project when talking into the microphone:

### Approach A: Browser Microphone STT $\rightarrow$ `/api/speak` $\rightarrow$ Voice Model TTS (Already Wired!)
Out of the box in `static/index.html`, clicking **`🎙️ Hold / Click to Talk`** uses the browser's `SpeechRecognition` API to transcribe your microphone speech into text and sends it to `POST /api/speak` in `server.py`.

To connect your coworker's Voice Model / TTS in `server.py`, edit `synthesize_voice_model_placeholder(text: str)`:

```python
def synthesize_voice_model_placeholder(text: str) -> tuple[str | None, float | None]:
    filename = f"speech_{uuid.uuid4().hex[:8]}.wav"
    filepath = os.path.join(AUDIO_CACHE_DIR, filename)

    # 1. Optional: Pass `text` to an LLM to generate a conversational reply first:
    # reply_text = my_llm.generate(text)

    # 2. Call your coworker's TTS / Voice Model to write `filepath` (.wav):
    # my_voice_model.synthesize_to_wav(text, output_path=filepath)

    # 3. Compute exact WAV duration so visemes & hand gestures match the audio 1:1:
    with wave.open(filepath, "rb") as wf:
        duration = wf.getnframes() / float(wf.getframerate())

    return f"/audio_cache/{filename}", duration
```

### Approach B: Raw Microphone Audio (`MediaRecorder`) $\rightarrow$ End-to-End Voice Model (Whisper / Speech-to-Speech)
If your coworker's voice pipeline takes **raw microphone audio** (`.webm` / `.wav`) directly:

1. **In `static/index.html`**, capture microphone audio with `navigator.mediaDevices.getUserMedia`:
   ```javascript
   const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
   const recorder = new MediaRecorder(stream);
   const chunks = [];
   recorder.ondataavailable = (e) => chunks.push(e.data);
   recorder.onstop = async () => {
     const audioBlob = new Blob(chunks, { type: 'audio/webm' });
     const formData = new FormData();
     formData.append('audio', audioBlob, 'mic.webm');

     // Send raw microphone audio to your backend voice pipeline
     const res = await fetch('/api/voice-chat', { method: 'POST', body: formData });
     const data = await res.json(); // returns { text, duration, audio_url, visemes }
     playAvatarResponse(data);
   };
   ```

2. **Real-Time Audio Frequency / Amplitude Lip-Sync (Optional Web Audio API Hook)**:
   If your voice model streams audio chunks via WebRTC or WebSockets without phoneme timestamps, you can drive `girl12.glb`'s morph targets directly from a Web Audio `AnalyserNode`:
   ```javascript
   const audioCtx = new AudioContext();
   const source = audioCtx.createMediaElementSource(audioElement);
   const analyser = audioCtx.createAnalyser();
   analyser.fftSize = 256;
   source.connect(analyser);
   analyser.connect(audioCtx.destination);
   const freqData = new Uint8Array(analyser.frequencyBinCount);

   // Inside requestAnimationFrame:
   analyser.getByteFrequencyData(freqData);
   const lowEnergy  = avg(freqData.slice(2, 10)) / 255.0;  // Vowels -> V_Open & Jaw
   const midEnergy  = avg(freqData.slice(10, 30)) / 255.0; // Spread -> V_Wide / V_Tight_O
   const highEnergy = avg(freqData.slice(30, 60)) / 255.0; // Fricatives -> V_Dental_Lip / V_Affricate
   setMorphTarget('V_Open', Math.min(1.0, lowEnergy * 1.4));
   setMorphTarget('V_Wide', Math.min(1.0, midEnergy * 1.1));
   setMorphTarget('V_Affricate', Math.min(1.0, highEnergy * 1.2));
   ```

---

## 4. How to Deploy on a Free Service

You can deploy this project for free in under 2 minutes using **Render**, **Hugging Face Spaces**, **Railway**, or **Vercel/Netlify**.

### Option 1: Deploy Full Stack (Python Backend + 3D Frontend) on Render.com (Free Tier)
1. Sign in to [Render.com](https://render.com) with your GitHub account.
2. Click **New +** $\rightarrow$ **Web Service** and select the **`xvimnt/UIUX_IA_2026`** repository.
3. Configure the service:
   - **Runtime**: `Python 3`
   - **Build Command**: `echo "No build dependencies required"` *(or `pip install -r requirements.txt` once you add your voice model)*
   - **Start Command**: `python3 server.py`
   - **Instance Type**: `Free`
4. Render automatically injects the `PORT` environment variable, which `server.py` reads (`PORT = int(os.environ.get("PORT", 8765))`).

### Option 2: Deploy on Hugging Face Spaces (Free CPU / ZeroGPU — Best for AI Voice Models!)
If your coworker's voice model uses PyTorch / Transformers / Kokoro / Whisper, **Hugging Face Spaces** gives you **16 GB RAM and 2 vCPUs for free**!
1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space) and choose **Docker** (Blank).
2. Add a 5-line `Dockerfile` to the root of this repository:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY . .
   ENV PORT=7860
   EXPOSE 7860
   CMD ["python3", "server.py"]
   ```
3. Push to your Hugging Face Space remote—it will serve the 3D avatar and the Python voice model on port `7860`.

### Option 3: Static Frontend on GitHub Pages / Vercel / Netlify
Because `static/index.html` also supports the browser's built-in `window.speechSynthesis` fallback, you can host the `static/` directory directly on **Vercel**, **Netlify**, or **GitHub Pages** for free, and point `fetch('https://your-voice-api.onrender.com/api/speak')` to your backend URL.
