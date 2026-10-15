# SpeechLens

**Contrastive speech analytics with temporally grounded, explainable delivery feedback.**
Multimodal AI Hackathon 2026, Track C: Contrastive Speech Analytics & Temporal Flaw Grounding.

SpeechLens compares a participant's recording of a passage against a reference recording of the same text, and tells the speaker exactly where their delivery deviates and why. Every detected flaw is tied to:

- a **timestamp** (start and end, in seconds) and, when forced alignment is available, the **word** it occurs after or spans,
- the **measurements** from both recordings,
- the **named rule** and threshold that triggered it,
- the **observed evidence**, kept separate from the **inferred interpretation**, and a recommendation.

There is no LLM anywhere in detection or scoring. Every number comes from signal processing on the audio, every threshold lives in a YAML file, and anything that cannot be measured is reported as `unavailable` with a reason instead of being guessed.

---

## Contents

- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the app](#running-the-app)
- [Using the app](#using-the-app)
- [Command-line tools](#command-line-tools)
- [Running the tests](#running-the-tests)
- [Configuration](#configuration)
- [API reference](#api-reference)
- [Project structure](#project-structure)
- [Design decisions](#design-decisions)
- [Limitations](#limitations)
- [Project status](#project-status)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## How it works

```
 Reference audio ─┐                                    ┌─► Flaws (timestamped, word-anchored)
                  ├─► Decode ─► Features ─► Alignment ─┼─► Rubric scores
 Participant ─────┘   16 kHz    RMS, VAD,   CTC word   └─► Dashboard
 audio + transcript   mono      pYIN F0,    timings
                                rate        (or energy fallback)
```

1. **Decode and validate.** WAV, MP3 and M4A are decoded with PyAV, downmixed to mono and resampled to 16 kHz with soxr. The original upload is kept byte-identical for playback. Files that are empty, corrupted, over 50 MB, shorter than 1 s or longer than 3 minutes are rejected with a specific error. No peak normalisation is applied, so loudness measurements stay meaningful.

2. **Extract features** on a shared 10 ms time grid:
   - **RMS energy** (25 ms window), reported both absolute and in dB relative to the recording's median speech level.
   - **Voice activity and pauses**, using an energy threshold relative to the recording's own level, so the same speech at a different volume yields identical pauses. Leading and trailing silence are reported separately and never counted as pauses.
   - **Pitch (F0)** with pYIN (64 ms window, 65–400 Hz). Unvoiced frames are `NaN`, never 0. A frame counts as voiced only if pYIN marks it voiced *and* the VAD marks it as speech.
   - **Speaking rate** (words per minute over the speaking span) and **articulation rate** (words per minute of actual speech, excluding pauses).

3. **Forced alignment.** If the optional `align` extra is installed, each word in the transcript is aligned to the audio using `facebook/wav2vec2-base-960h` CTC emissions and a Viterbi trellis implemented in NumPy in this repo. Each word gets a start, end and alignment score (mean frame log-probability along the path), at 20 ms resolution. Without the extra, the pipeline runs in **energy fallback** mode, which gives pauses and recording-level rates but no word timings. The mode used is reported in every result and shown in the UI.

4. **Compare and detect flaws.** Words are matched between the two recordings **by transcript position, never by time**, so recordings of different lengths are compared correctly. Four rules run (see [Configuration](#configuration)):

   | Rule | Scope | What it compares |
   |---|---|---|
   | `long_pause` | per pause | Participant pause after word *i* vs the reference pause after word *i* (with alignment); vs the reference's pause distribution (fallback) |
   | `local_rate` | word window | Participant vs reference onset-to-onset span over a sliding window of words (requires alignment) |
   | `rate_deviation` | recording | Participant / reference articulation rate |
   | `flat_pitch` | recording | Participant / reference F0 spread (interquartile range in semitones, relative to each speaker's own median) |

   Words with a poor alignment score are excluded from word-level comparison. If too many are excluded, word-level rules return `unavailable` and the result warns that the transcript may not match the audio.

5. **Score.** A deterministic rubric scores **pacing**, **pause control** and **vocal variation** from 0 to 100, and combines them as a weighted mean. Unavailable dimensions are dropped and the remaining weights renormalised, and the result says so.

---

## Requirements

| | Version | Notes |
|---|---|---|
| Python | 3.13 | Installed and managed by `uv` |
| [uv](https://docs.astral.sh/uv/) | recent | Python package and environment manager |
| Node.js | 20 or later | For the frontend |
| Disk | ~0.4 GB core, ~3 GB with alignment | PyTorch is the bulk of it |
| GPU | Optional | Alignment runs on CPU too; a CUDA GPU makes it several times faster |

Developed and tested on Windows 11 with an NVIDIA RTX 3050 (4 GB VRAM) and 8 GB RAM. Peak VRAM for alignment stayed under 1 GB on a 3-minute file. The commands below are for Windows Command Prompt; on macOS or Linux, use `/` instead of `\` in paths.

> **Note on the `align` extra:** `pyproject.toml` pins `torch==2.7.1+cu126` from the PyTorch CUDA 12.6 index. On a machine without an NVIDIA GPU, or on macOS, this pin will not resolve; either use the core install (energy fallback) or edit the `align` extra to use the CPU build of torch.

---

## Installation

Clone the repository:

```
git clone <repository-url> speechlens
cd speechlens
```

**Backend.** Choose one:

```
cd backend

:: Core install: everything except word-level alignment (~0.4 GB)
uv sync --extra dev

:: Full install: adds forced alignment via PyTorch + transformers (~3 GB)
uv sync --extra dev --extra align
```

The first analysis with the full install downloads the wav2vec2 model (~360 MB) from Hugging Face and caches it.

**Frontend:**

```
cd frontend
npm install
```

---

## Running the app

Use two terminals and keep both open.

**Terminal 1: backend**

```
cd backend
uv run uvicorn app.main:app --port 8000
```

Wait for `Uvicorn running on http://127.0.0.1:8000`. You can confirm it is up at http://localhost:8000/api/health.

**Terminal 2: frontend**

```
cd frontend
npm run dev
```

Wait for `Local: http://localhost:5173/`, then open **http://localhost:5173** in your browser. The frontend proxies `/api` requests to the backend on port 8000.

To stop, press `Ctrl+C` in each terminal.

---

## Using the app

1. **Upload a reference recording.** This is the delivery you want to compare against.
2. **Upload a participant recording** of the same text.
3. **Paste the transcript** exactly as read. With forced alignment, the transcript must use letters only: write numbers as words ("twenty", not "20"). Punctuation is ignored.
4. Click **Run analysis**. Progress shows the real pipeline stage: decoding, extracting features, loading the alignment model, aligning, comparing, scoring.
   The first analysis after starting the backend takes one to two minutes while the alignment model loads; later analyses take seconds.
5. The **results page** shows:
   - the **overall score** and the **alignment mode** used,
   - **both recordings** with waveforms; flaw regions are highlighted on the participant waveform,
   - the **transcript**, with each word clickable to seek playback and words inside flaws highlighted (alignment mode only),
   - the **rubric breakdown**, with each dimension's score, weight and inputs,
   - the **flaw list**: severity, time range, evidence, interpretation, measurements and recommendation. Click a flaw to jump to it,
   - an **unavailable** panel listing anything that could not be measured, and why.

### Trying it without your own recordings

On Windows, generate synthetic test audio with the built-in speech synthesiser:

```
powershell -ExecutionPolicy Bypass -File scripts\make_dev_fixture.ps1
```

This writes `tts_ideal.wav`, `tts_pauses.wav` (three inserted 2-second pauses), `tts_fast.wav` and `passage.txt` to `backend\dev_fixtures\`. Use `tts_ideal.wav` as the reference, either of the others as the participant, and the contents of `passage.txt` as the transcript.

These fixtures are for development only. A synthetic voice has unnaturally regular pitch and timing, so they are never used to tune thresholds or reported as evaluation results.

---

## Command-line tools

Run these from the `backend` folder so they use the project environment.

**Inspect the features of one recording:**

```
uv run python ..\scripts\inspect_features.py <audio_file> <transcript_file>
```

Prints duration, leading and trailing silence, every pause, speech time, speaking and articulation rate, F0 statistics, voiced fraction and extraction time.

**Compare two recordings from the command line:**

```
uv run python ..\scripts\run_comparison.py <reference_audio> <participant_audio> <transcript_file>
```

Prints the full result as JSON: every flaw, every rubric dimension with its inputs, and the unavailable list.

**Run forced alignment on its own** (requires the `align` extra):

```
uv run python ..\scripts\ctc_spike.py <audio_file> <transcript_file> [--device cuda|cpu] [--vad-check] [--repeat N]
```

Prints word timings and scores, checks that every word is present and times are monotonic, and reports timing and memory use. `--vad-check` compares word boundaries against VAD-detected pauses; `--repeat N` concatenates the audio N times as a stress test.

---

## Running the tests

```
cd backend
uv run pytest tests -v
```

Tests cover audio decoding and validation, feature extraction (against signals with known properties: tones of known frequency, silences at known times, gain changes), the CTC trellis on hand-built emission matrices, the rules and rubric, and the API end to end.

Tests that need the alignment model are skipped automatically when the `align` extra is not installed, so the core install runs the rest without PyTorch.

---

## Configuration

All thresholds, windows and weights are in `backend/config/`, not in code. Changing a YAML value changes behaviour without code changes; restart the backend afterwards.

| File | Contents |
|---|---|
| `features.yaml` | Sample rate, upload limits, frame and hop sizes, VAD threshold, minimum pause length, F0 search range |
| `rules_v1.yaml` | Each rule's thresholds, severity bands and recommendation text; word-comparison exclusion settings |
| `rubric_v1.yaml` | Each dimension's formula parameters and weight |

Key defaults:

| Setting | Value | Meaning |
|---|---|---|
| Analysis rate / hop | 16 kHz / 10 ms | Shared time grid for all features |
| VAD threshold | −20 dB | Relative to the recording's median RMS |
| Minimum pause | 0.25 s | Shorter silences are not counted as pauses |
| `long_pause` threshold | max(1.5 s, 2 × reference median pause) | |
| `rate_deviation` band | 0.7–1.3 × reference | Articulation-rate ratio |
| `local_rate` band | 0.6–1.4 × reference | Over a 5-word window |
| `flat_pitch` | participant F0 spread < 0.5 × reference | Needs ≥ 50 voiced frames in each recording |
| Word exclusion | alignment score < −1.0 | Word-level rules unavailable if > 30% of words excluded |
| Rubric weights | pacing 0.35, pause control 0.35, vocal variation 0.30 | |

**These values are provisional.** They were set from first principles, not tuned on data, and are marked as such in the files.

### Rubric formulas

| Dimension | Formula |
|---|---|
| Pacing | `max(0, 100 − 250 × |articulation-rate ratio − 1|)` |
| Pause control | `max(0, 100 − Σ penalties)`, with each `long_pause` flaw costing 8 (mild), 20 (moderate) or 40 (severe) |
| Vocal variation | `100 × clamp(participant F0 IQR / reference F0 IQR, 0, 1)` |
| Overall | Weighted mean of available dimensions, weights renormalised |

---

## API reference

The backend serves a JSON API on port 8000. Interactive documentation is generated automatically at http://localhost:8000/docs.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Liveness check |
| `POST` | `/api/recordings` | Upload an audio file (multipart field `file`). Returns `id`, `duration`, `sample_rate`, `warnings` |
| `POST` | `/api/analyses` | Start an analysis. Body: `participant_id`, `reference_id`, `transcript`, optional `rubric_version`. Returns the analysis `id` |
| `GET` | `/api/analyses/{id}` | Status, current `stage`, and the full `result` once complete |
| `GET` | `/api/recordings/{id}/audio` | The original uploaded audio, for playback |
| `GET` | `/api/rubric` | Active rubric dimensions and rule thresholds |

Analyses run in the background. Poll `GET /api/analyses/{id}` until `status` is `complete` or `error`.

Each flaw in a result contains: `flaw_id`, `rule_id`, `category`, `severity`, `start_seconds`, `end_seconds`, `reference_measurement`, `participant_measurement`, `deviation`, `evidence`, `interpretation` and `recommendation`.

---

## Project structure

```
backend/
  app/                    FastAPI app: routes, request/response schemas, storage, pipeline runner
  speechlens/
    audio/                Decoding, validation, resampling
    features/             RMS, VAD and pauses, pYIN F0, speaking rate
    alignment/            Aligner interface, CTC aligner (wav2vec2 + NumPy trellis), energy fallback
    flaws/                Rule engine
    rubric/               Rubric scoring
  config/                 features.yaml, rules_v1.yaml, rubric_v1.yaml
  tests/                  pytest suite
  dev_fixtures/           Generated synthetic audio (not committed)
frontend/
  src/pages/              Workspace (upload and run) and Results pages
  src/api.ts              Typed API client
scripts/                  Command-line tools and fixture generator
dataset/                  Contrastive dataset: audio, transcripts, annotations
docs/                     Project documentation
```

Uploaded audio and analysis results are stored in `backend/uploads/` and `backend/analysis_outputs/`. Both are excluded from git and can be deleted at any time.

---

## Design decisions

- **Rules, not a trained model.** With a small dataset, a trained classifier would overfit and could not explain itself. Transparent, configurable rules make every detection auditable.
- **Words are matched by transcript position, never by time.** Two readings of the same text differ in length and pacing; comparing "the same second" in each would compare different words.
- **No time warping on timing features.** Pauses and rate are compared in word-indexed space. Warping the recordings onto each other first would erase the very pacing differences the system is meant to detect.
- **Speaker-relative pitch.** Pitch spread is measured in semitones relative to each speaker's own median, so a higher or lower voice is not mistaken for more or less variation.
- **Gain-invariant measurements.** The VAD threshold and loudness values are relative to each recording's own level, so microphone gain does not change the result.
- **Pause boundaries come from the VAD, not from word gaps.** In testing, CTC word *starts* landed within about 30 ms of measured pause edges, but word *ends* came 20–190 ms early. Pause edges therefore come from the energy detector, and local pacing uses onset-to-onset intervals rather than word durations.
- **The alignment model loads once** per server process, on first use, and is reused.
- **Evidence is separated from interpretation.** "This pause is 2.99 s; the reference pause at the same point is 1.01 s" is a measurement. "This may interrupt the flow between ideas" is an inference. They are stored and displayed separately.

---

## Limitations

- **Thresholds are untuned.** All rule thresholds and rubric weights are provisional values that have not yet been calibrated on human recordings.
- **Each long pause is also flagged as a local-rate flaw.** A long pause stretches the onset-to-onset span of the window around it, so `local_rate` fires on the same region as `long_pause`. Similarly, a uniformly fast recording is flagged as locally fast across its whole length. These overlapping detections are not yet de-duplicated.
- **The rubric's pause-control score counts only `long_pause` flaws**, and pacing uses only the recording-level rate ratio; `local_rate` flaws are reported but do not affect the score.
- **Alignment vocabulary is English letters only.** Digits and symbols in the transcript raise an error naming the offending words.
- **Fillers ("um", "uh") and repetitions are not detected.** Forced alignment to a clean transcript cannot see words that are not in it. This was out of scope for the deadline.
- **Acoustic features do not measure confidence or persuasiveness.** A flat pitch contour is a measurable property, not proof that a speaker lacks confidence. Interpretations are worded as possibilities for this reason.
- **Analyses are stored in memory and as JSON files.** Restarting the backend clears the in-memory list of analyses.

---

## Project status

| Component | Status |
|---|---|
| Audio decoding and validation | Done, tested |
| Feature extraction (RMS, VAD, pauses, F0, rate) | Done, tested |
| Forced alignment (wav2vec2 + NumPy trellis) | Done, tested; integrated into the pipeline |
| Rules: long pause, local rate, rate deviation, flat pitch | Done, tested |
| Rubric v1 | Done, tested |
| REST API | Done, tested |
| Web app: upload, progress, results, waveform, clickable transcript | Done |
| Contrastive dataset of human recordings | **Not yet in the repository** |
| Evaluation (precision/recall/F1, temporal IoU) | **Not yet done** |
| Reference-vs-participant feature plots, dataset browser | Not yet done |

So far, the pipeline has been validated on signals with known properties and on synthetic speech. No accuracy figures are reported, because there is not yet a labelled set of human recordings to measure them on.

---

## Troubleshooting

**"Port 8000 is already in use", or the app shows stale results after a code change.**
An earlier backend process is still running. Close the backend terminal, then in a new Command Prompt run:

```
taskkill /F /IM python.exe
```

This also stops any other Python program you have open. Then start the backend again. Avoid `--reload` on Windows; it has been seen to leave processes behind.

**Results say "alignment: energy fallback".**
The `align` extra is not installed in the backend environment. Run `uv sync --extra dev --extra align` in `backend`, then restart the backend.

**"Transcript contains characters outside the model vocabulary".**
Write numbers and symbols as words, e.g. "twenty five percent" instead of "25%".

**Word-level rules are unavailable with a transcript-mismatch warning.**
More than 30% of words aligned poorly. Check that the transcript matches what was actually said.

**The first analysis is very slow.**
Expected: the alignment model is loading (and, the very first time, downloading). Later analyses reuse it.

**`npm run dev` works but the page shows errors talking to the API.**
The backend is not running, or is running on a port other than 8000.

---

## License

MIT.
