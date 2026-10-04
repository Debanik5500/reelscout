"""Local speech-to-text via faster-whisper (free, CPU)."""
from __future__ import annotations
import threading

_MODELS: dict = {}

# Biasing the decoder toward tech vocabulary noticeably helps with tool / domain names.
PROMPT = ("A short social-media video about software, startups and AI tools. It may mention "
          "websites like acquire.com, github.com/owner/repo, Claude, Cursor, Vercel, Supabase, "
          "Hugging Face, Notion.")

# Pre-load default model at import time for fast first inference
_DEFAULT_MODEL = None
_warmup_done = False


def _get_model(model_size: str = "tiny"):
    from faster_whisper import WhisperModel
    if model_size not in _MODELS:
        _MODELS[model_size] = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            cpu_threads=4,
            num_workers=2
        )
    return _MODELS[model_size]


def transcribe(wav_path: str, model_size: str = "tiny") -> list[dict]:
    """Transcribe audio using faster-whisper.
    
    Uses 'tiny' model by default for speed (~0.3s on CPU).
    Use model_size='small' for better accuracy (slower).
    """
    model = _get_model(model_size)
    segs, _ = model.transcribe(
        wav_path,
        beam_size=1,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        initial_prompt=PROMPT,
        condition_on_previous_text=False,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6
    )
    return [dict(start=round(s.start, 2), end=round(s.end, 2), text=s.text.strip()) for s in segs]


# Background warmup for instant first inference
def _warmup():
    try:
        _MODELS["tiny"].transcribe(
            "/dev/null", beam_size=1, vad_filter=False, condition_on_previous_text=False
        )
    except Exception:
        pass


threading.Thread(target=_warmup, daemon=True).start()