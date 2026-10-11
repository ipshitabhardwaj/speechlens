# SpeechLens — CLAUDE.md

## What this project is
SpeechLens: contrastive speech analytics tool for the Multimodal AI Hackathon 2026, Track C.
Compares a participant recording against a reference recording of the same text, detects delivery flaws,
and grounds each one with timestamp, measurement, named rule, and plain explanation.

Full kickoff prompt: [docs/kickoff.md](docs/kickoff.md).

## Project facts
- Deadline: 15 Oct 2026. Team: 2 people.
- Dataset speakers: both team members. English only.
- Repo licence: MIT. No GPLv3 dependencies (this excludes praat-parselmouth).

## Machine constraints
- Windows x64, RTX 3050 (4 GB VRAM), 8 GB RAM.
- Everything runs locally. No paid APIs. No LLM in scoring or detection path.
- Python 3.13 (librosa 1.0 supports it; 3.12 not installed on this machine). Use `uv` for dependency management, pin versions.

## Non-negotiable rules
1. Never fabricate measurements, scores, timestamps, accuracy numbers, dataset sizes, or chart data.
   If something can't be computed, return `unavailable` with a reason.
2. Every flaw carries: start/end seconds, category, severity, reference and participant measurements,
   deviation, rule_id, evidence, interpretation, recommendation.
   Keep observed evidence separate from inferred listener impact.
3. No fake buttons, placeholder charts, or TODO stubs for essential features.
   Ship a simpler working version and document the limitation instead.
4. Never claim forced alignment happened when the fallback ran.
   The alignment method is reported in every result and shown in the UI.
5. A deviation from the reference is not automatically a flaw. Rules score defined criteria only.
6. Don't say something works until you have run it. Show command and output.

## Architecture decisions (do not change without discussing first)
- **Audio decode**: PyAV + soundfile (no system ffmpeg). Keep original for playback; analyse 16 kHz mono copy.
  Never peak-normalise the analysis copy before measuring energy.
- **Features**: librosa for STFT, MFCC, RMS, pYIN F0 with voiced flags.
  Configurable frame/hop sizes in `backend/config/features.yaml`, units documented.
- **Alignment**: adapter interface with two implementations:
  - `energy_fallback`: VAD-based pauses and overall rate only, no word times.
  - `ctc`: wav2vec2-base-960h loaded via transformers (Apache-2.0), emissions + NumPy Viterbi trellis,
    per-word start/end, score = mean frame log-prob. Chunk long audio; cap uploads at 3 minutes.
  WhisperX is excluded (it pulls pyannote-audio and pins torch 2.8).
- **Comparison**: match words by transcript index, never by time. Timing features (word durations,
  pauses, local rate) are compared in word-indexed space with no time warping, so pacing flaws are
  never warped away. Pitch in semitones relative to speaker's median; energy in dB relative to
  recording's median speech level. Skip pitch rules when too few voiced frames.
  Contour DTW inside matched phrases is CUT for the 15 Oct deadline; do not build it.
- **Fillers**: CUT for the 15 Oct deadline; do not build. (Design: voiced stretches between aligned
  words unexplained by transcript; labelled lower-reliability.)
- **Rules/rubric**: versioned YAML in `backend/config/`, pure deterministic functions,
  thresholds and weights never hardcoded in application code.
- **Backend**: FastAPI, Pydantic schemas, files on disk, JSON for metadata (no SQLite).
- **Frontend**: React + Vite + TypeScript + Tailwind, WaveSurfer.js, Recharts.
  Warm off-white background, charcoal/navy text, muted coral for flaws, restrained green/blue for comparisons.
  No gradients or glassmorphism.
- **faster-whisper** (base/small) optional, only for transcript drafting and mismatch flagging.

## Repository layout
```
speechlens/
  backend/app/                   routes, schemas, job status
  backend/speechlens/{audio,features,alignment,compare,flaws,rubric,dataset}/
  backend/config/                rubric_v1.yaml, rules_v1.yaml, features.yaml
  backend/tests/
  frontend/
  dataset/                       manifest.jsonl, schema.json, annotations/, audio/
  scripts/                       build_dataset, evaluate, stress_test
  docs/                          architecture, dataset, rubric, evaluation, technical_report
```

## API endpoints
```
POST /api/recordings              upload; returns id, duration, sample rate, warnings
POST /api/analyses                participant id, reference id, transcript, rubric version -> job id
GET  /api/analyses/{id}           real stage status, then full result
GET  /api/recordings/{id}/audio   original audio
GET  /api/dataset/pairs
GET  /api/dataset/pairs/{pair_id}
GET  /api/rubric                  active rules, thresholds, weights
```

## Commands
```bash
# Backend
cd backend
uv run uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev

# Tests
cd backend
uv run pytest tests/ -v
```

## Dataset rules
- One JSONL record per recording: recording_id, speech_id, pair_id, reference_recording_id,
  speaker_id, transcript, condition (ideal | mild_flaw | moderate_flaw | severe_flaw),
  flaw_categories, origin (human_recorded | synthetic_transform), transform {type, params,
  source_recording_id}, audio_path, word_alignments_path, flaw_segments [{category, start_seconds,
  end_seconds, severity}], annotation_method, recording_metadata, provenance {recorded_by, licence,
  consent}, split. Full schema: [docs/kickoff.md](docs/kickoff.md).
- Passages are public-domain text recorded by our team. No third-party speech audio.
- Splits assigned per speech_id so a passage never crosses splits.
- Synthetic transforms record exact ground-truth boundaries; always labelled and evaluated separately.

## Dependency policy
- Before adding any dependency, check it installs on Windows with Python 3.12; report size and licence.
- No GPLv3 dependencies (repo is MIT). This excludes praat-parselmouth.
- torch and transformers are an optional extra named `align` (`uv sync --extra align`).
  The core pipeline with energy_fallback installs and runs without them.
- All deps pinned in pyproject.toml / package.json.

## How we work
- **Phase 1** (current): working vertical slice — energy_fallback aligner, three rules, end-to-end UI.
  CTC aligner spike runs in parallel as a half-day timebox; if it fails, stop and tell me before
  choosing a fallback.
- **Phase 2**: integrate CTC aligner into the pipeline; word-level timestamps in results and UI;
  word-indexed comparison of word durations and pauses; local speaking-rate rule; per-word alignment
  scores with low-score words excluded from comparison.
- **Phase 3**: recorded dataset, annotations, evaluation (F1, temporal IoU, repeatability).
- **Phase 4**: synchronised plots, transcript highlighting, minimal dataset browser (list of dataset
  pairs showing condition and origin labels; selecting one loads it into the results view).
- **Phase 5**: requirement audit, six-page technical report, demo checklist.
- **Cut** (deadline 15 Oct): filler detection, contour DTW, full dataset explorer page, practice retakes.
- Stop at the end of each phase: report what works (with commands run), files changed,
  how to run and test, what is incomplete, and the next step. Then wait for approval.
- Never run git commit, git push, git add, or any command that changes git history or the index.
  I do all commits myself. At each natural checkpoint, stop and tell me: the files changed,
  a suggested commit message, and the tests that pass. You may run read-only git commands
  (status, diff, log).
- Tests written alongside each module, not afterwards.
- Ask only what can't be determined by inspecting the repo or running code.
