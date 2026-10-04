from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path

# One OpenAI-compatible client covers every preset below except Anthropic's native API.
# Model names are defaults only -- override with REELSCOUT_MODEL if a provider renames one.
PRESETS = {
    "gemini": dict(base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                   model="gemini-2.5-flash", key_env="GEMINI_API_KEY"),
    "groq": dict(base_url="https://api.groq.com/openai/v1",
                 model="llama-3.3-70b-versatile", key_env="GROQ_API_KEY"),
    "openrouter": dict(base_url="https://openrouter.ai/api/v1",
                       model="meta-llama/llama-3.3-70b-instruct:free", key_env="OPENROUTER_API_KEY"),
    "openai": dict(base_url="https://api.openai.com/v1",
                   model="gpt-4.1", key_env="OPENAI_API_KEY"),
    "xai": dict(base_url="https://api.x.ai/v1",
                model="grok-beta", key_env="XAI_API_KEY"),
    "ollama": dict(base_url="http://localhost:11434/v1", model="qwen2.5:3b", key_env=""),
    "anthropic": dict(base_url="https://api.anthropic.com",
                      model="claude-haiku-4-5-20251001", key_env="ANTHROPIC_API_KEY"),
}


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip())


@dataclass
class Settings:
    llm: str = "none"
    model: str = ""
    vision_model: str = ""        # optional separate model for frame calls (e.g. a local VLM)
    whisper_model: str = "tiny"
    max_frames: int = 16
    timeout: int = 0              # seconds per LLM call; 0 = auto (600 for local Ollama, 120 for hosted)
    cache_dir: Path = Path(".cache")
    vision: str = "auto"          # off | auto (only when text signals are weak) | on (always)
    vision_frames: int = 6
    weak_below: float = 0.65
    resolve: str = "off"          # off | auto (when still weak and there are clues)
    image_search: bool = False    # PicImageSearch (scrapes Google Lens/Yandex/Bing) -- fragile, optional
    geo: bool = False             # GeoCLIP region-level geolocation -- needs torch, optional
    verify: bool = True
    text_policy: str = "auto"     # auto: skip the TEXT LLM call when local evidence is already decisive | always | never
    # Frame sampling
    sample_hz: float = 2.0        # change-aware sampling rate (fps)
    fixed_hz: float = 1.0         # fixed-interval sampling (e.g., 1.0 = 1 fps); 0 = use change-aware
    # OCR parallelism
    ocr_workers: int = 4          # >1 enables parallel OCR across frames

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        return cls(
            llm=os.getenv("REELSCOUT_LLM", "none").lower(),
            model=os.getenv("REELSCOUT_MODEL", ""),
            vision_model=os.getenv("REELSCOUT_VISION_MODEL", ""),
            timeout=int(os.getenv("REELSCOUT_TIMEOUT", "0") or 0),
            text_policy=os.getenv("REELSCOUT_TEXT_POLICY", "auto").lower(),
            whisper_model=os.getenv("REELSCOUT_WHISPER", "tiny"),
            sample_hz=float(os.getenv("REELSCOUT_SAMPLE_HZ", "1.0") or 1.0),
            fixed_hz=float(os.getenv("REELSCOUT_FIXED_HZ", "1.0") or 1.0),
            ocr_workers=int(os.getenv("REELSCOUT_OCR_WORKERS", "4") or 4),
            max_frames=int(os.getenv("REELSCOUT_MAX_FRAMES", "10") or 10),
        )

    def for_stage(self, stage: str) -> "Settings":
        """Per-stage provider/model override, e.g. REELSCOUT_RESEARCH_LLM=openai REELSCOUT_RESEARCH_MODEL=gpt-4.1.
        Stages: TEXT (read the evidence), VISION (look at frames), RESEARCH (turn clues into a name).
        With no override a stage uses the main REELSCOUT_LLM / REELSCOUT_MODEL."""
        key = f"REELSCOUT_{stage.upper()}"
        llm, model = os.getenv(key + "_LLM", "").lower(), os.getenv(key + "_MODEL", "")
        if not llm and not model:
            return self
        same = llm in ("", self.llm)
        # switching provider without naming a model means "that provider's default", never the main stage's model
        return dataclasses.replace(self, llm=llm or self.llm, model=model or (self.model if same else ""),
                                   vision_model=self.vision_model if same else "")

    @property
    def preset(self) -> dict | None:
        return PRESETS.get(self.llm)

    @property
    def api_key(self) -> str:
        p = self.preset
        return os.getenv(p["key_env"], "") if p and p["key_env"] else ""
