import os
import uuid
import io
import torch
import torchaudio
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VOICES = Path("/tmp/voices")
VOICES.mkdir(exist_ok=True)

_ready = False
_converter = None
_tts = None
_source_se = None


def load_models():
    global _ready, _converter, _tts, _source_se
    if _ready:
        return
    print("Loading OpenVoice models...")
    from openvoice.api import ToneColorConverter
    from melo.api import TTS
    from huggingface_hub import snapshot_download

    ckpt = snapshot_download(repo_id="myshell-ai/openvoice-v2")

    _converter = ToneColorConverter(f"{ckpt}/converter/config.json", device="cpu")
    _converter.load_ckpt(f"{ckpt}/converter/checkpoint.pth")

    _tts = TTS(language="ZH", device="cpu")

    # Base speaker for ZH
    se_path = f"{ckpt}/base_speakers/ses/zh.pth"
    _source_se = torch.load(se_path, map_location="cpu")

    _ready = True
    print("Models ready.")


@app.on_event("startup")
async def startup():
    load_models()


@app.get("/health")
def health():
    return {"status": "ok", "ready": _ready}


@app.post("/clone")
async def clone_voice(audio: UploadFile = File(...)):
    """Upload a reference audio, returns voice_id for future synthesis."""
    load_models()
    from openvoice import se_extractor

    voice_id = str(uuid.uuid4())[:8]
    ref_path = VOICES / f"{voice_id}_ref.wav"

    data = await audio.read()
    audio_io = io.BytesIO(data)

    try:
        waveform, sr = torchaudio.load(audio_io)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 22050:
            waveform = torchaudio.transforms.Resample(sr, 22050)(waveform)
        torchaudio.save(str(ref_path), waveform, 22050)
    except Exception as e:
        raise HTTPException(400, f"Audio error: {e}")

    try:
        target_se, _ = se_extractor.get_se(
            str(ref_path), _converter, vad=True
        )
        torch.save(target_se, VOICES / f"{voice_id}_se.pth")
    except Exception as e:
        raise HTTPException(500, f"Voice extraction failed: {e}")

    return {"voice_id": voice_id}


class SynthRequest(BaseModel):
    text: str
    voice_id: str
    speed: float = 1.0


@app.post("/synthesize")
async def synthesize(req: SynthRequest):
    """Synthesize text with cloned voice, returns WAV audio."""
    load_models()

    se_path = VOICES / f"{req.voice_id}_se.pth"
    if not se_path.exists():
        raise HTTPException(404, "Voice not found. Please re-register.")

    target_se = torch.load(str(se_path), map_location="cpu")

    base_wav = f"/tmp/{uuid.uuid4()}_base.wav"
    out_wav = f"/tmp/{uuid.uuid4()}_out.wav"

    try:
        spk_id = _tts.hps.data.spk2id["ZH"]
        _tts.tts_to_file(req.text, spk_id, base_wav, speed=req.speed)

        _converter.convert(
            audio_src_path=base_wav,
            src_se=_source_se,
            tgt_se=target_se,
            output_path=out_wav,
        )

        with open(out_wav, "rb") as f:
            audio_bytes = f.read()
    finally:
        for p in [base_wav, out_wav]:
            if os.path.exists(p):
                os.remove(p)

    return Response(content=audio_bytes, media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
