# reelscout

Find the **website / GitHub repo / product / place** a reel is about, even when the creator never says it.

## How it works (cheapest signal first)

| # | Stage | Cost | Why |
|---|-------|------|-----|
| 1 | Ingest (yt-dlp → instaloader fallback) | free | caption, @mentions, tagged accounts, location tag, creator's own comments |
| 2 | Frames, change-aware (OpenCV) | free | screen recordings are static; keep only frames that changed |
| 3 | Speech → text (faster-whisper, local) | free | spoken "acquire dot com" is normalised to `acquire.com` |
| 4 | OCR incl. 3x-zoomed top/bottom strips | free | URL bars are tiny; RapidOCR if installed, else Tesseract |
| 5 | Regex / social-signal extraction | free | URLs, github.com/x/y, @brands, `pip/npm/brew install x` commands, "comment LINK" detection |
| 6 | **One** text LLM call (Gemini/Groq/OpenRouter/Ollama free tiers) | ~free | fixes mishearings ("clawed"→Claude), lists entities, records `unresolved_clues` |
| 7 | Verification: HTTP check, GitHub API, PyPI/npm/brew/crates, OSM Nominatim, DDG | free | dead/hallucinated domains get penalised; corrected via `alternates` |
| 8 | **Auto vision tier**: up to 8 frames (1024px) in ONE vision-LLM call | small | runs only if nothing scored >= 0.65 after stages 1-7; `--vision off/auto/on` |

Nothing is lost if the LLM is down/rate-limited: stages 1-5 + 7 still return results.

## Stage 9: turning the unresolved subject into a name (all open source / free)

When the text and frame passes still leave nothing convincing, the subject is identified **without reverse image search**:

| Path | For | Notes |
|------|-----|-------|
| **A. Visual identification** (`visualid.identify`, default) | anything the frames show but nobody says | vision-LLM pass describes the frames as a ranked *visual fingerprint* (distinctive features, high/medium/low) + hypotheses — caption, transcript and on-screen text are clues, not ground truth; ordinary **text** search is then driven by combinations of the rarest features (alternate names, local terminology, other languages, technical terms) with the transcript as context; a second LLM pass cross-checks candidates feature-by-feature, seeks disconfirmation, and assigns CONFIRMED / HIGHLY LIKELY / LIKELY / POSSIBLE / UNKNOWN |
| **B. Clue resolve** (`resolve.resolve`, fallback) | `unresolved_clues` → SearXNG / `ddgs`, GitHub search API, OSM Nominatim | runs when visual identification found nothing and there are clues (or the reel is gated, `--image-search` or `--geo`) |

The final LLM step may only return what the search results support; "unknown" is an allowed answer.

## Run

```bash
pip install -r requirements.txt && sudo apt install ffmpeg tesseract-ocr   # or: brew install ffmpeg tesseract
cp .env.example .env                                                        # default is fully local via Ollama
python -m reelscout my_reel.mp4 --llm ollama
python -m reelscout "https://www.instagram.com/reel/XXXX/" --cookies-from-browser chrome --llm gemini
python -m reelscout my_reel.mp4 --llm none            # zero-API mode (regex + verification only)
python -m reelscout my_reel.mp4 --llm gemini --vision on   # always add the frame pass (default: auto)
```

With an OpenAI key: put `OPENAI_API_KEY=...` and `REELSCOUT_LLM=openai` in `.env` (or `--llm openai`).
Going to production = `REELSCOUT_LLM=anthropic` (or any other provider); no code changes.
Stage outputs are cached in `.cache/` so you can iterate on prompts without re-running Whisper/OCR.

## Tests
`pytest tests` (fixtures come from your real SaaS-ideas reel).

## Local first, OpenAI for the hard parts

Always local and free: ingestion, frame sampling, Whisper, OCR, regex/@mention/install-command extraction, ASR-vs-screen spelling fold, verification (HTTP, GitHub, PyPI/npm, OpenStreetMap), scoring.
Sent to the model (gpt-4.1 by default): the first LLM call **only when** the local result isn't decisive (best candidate >= 0.65, verified, backed by two independent signals such as spoken + on screen), then frames and research only if still weak.
Override with `REELSCOUT_TEXT_POLICY=always|never` or `--text-llm`. `python -m reelscout.evaluate ... --ablate` compares local-only / hybrid / LLM-every-time on your labelled reels.

## Where the LLM is used (and where it isn't)

Whisper (speech) and OCR are always local and never touch an LLM. The model provider (OpenAI / Ollama / ...) is used in at most 4 calls per reel:

| Call | Stage name | When | Code |
|------|-----------|------|------|
| 1. read the evidence, fix mishearings, list entities | `TEXT` | unless local evidence is already decisive (see above) | `reason.reason` |
| 2. same, plus up to 8 frames | `VISION` | only if call 1 + verification left nothing >= 0.65 | `reason.reason(images=...)` |
| 3. describe frames as a visual fingerprint + hypotheses + queries | `VISION` | only if still weak and frames exist | `visualid.identify` (pass 1) |
| 4. cross-check search hits against the fingerprint | `RESEARCH` | right after 3, if search returned hits | `visualid.identify` (pass 2) |
| 5. write search queries from the clues | `RESEARCH` | only if visual identification found nothing and there are clues (or the reel is gated) | `resolve.make_queries` |
| 6. pick an answer from search results / page text | `RESEARCH` | right after 5 | `resolve.resolve` |

Verification (HTTP / GitHub / PyPI / OpenStreetMap), regex, search and image search make no LLM calls. The CLI prints one `llm[stage]` line per stage with the exact `provider:model` and token counts.
Each stage can use a different provider/model via `REELSCOUT_TEXT_*`, `REELSCOUT_VISION_*`, `REELSCOUT_RESEARCH_*` (see `.env.example`).

## Slow on a laptop?

- `python -m reelscout reel.mp4 --llm none` is instant once Whisper/OCR are cached: regex + verification only. Run this first.
- Use a small local model: `ollama pull qwen2.5:3b`, set `REELSCOUT_MODEL=qwen2.5:3b` in **your `.env`** (it overrides the defaults), and `export OLLAMA_KEEP_ALIVE=30m` before starting Ollama so it stays loaded between runs.
- If Ollama is down or the model isn't pulled, the tool now says so immediately instead of timing out. `REELSCOUT_TIMEOUT` raises the per-call limit.
- `--whisper base` is ~3x faster than `small` (less accurate on brand names). Stage results are cached in `.cache/`.
- `ddgs` needs Python 3.10+; on 3.9 web search falls back to the plain DuckDuckGo scraper.

## Evaluate before adding anything

1. Copy `eval/labels.example.csv`, label 25-30 real reels (bucket A-E, see `reelscout/evaluate.py`).
2. `python -m reelscout.evaluate eval/labels.csv --llm ollama --ablate`
3. The report shows top-1 / top-3 / abstain / **wrong-but-confident** per bucket, plus how much vision + resolve add over the cheap text-only path.
Add features only for the buckets that fail. If a stage adds ~0 in the ablation, delete it.
