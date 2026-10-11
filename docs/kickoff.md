Historical kickoff prompt. CLAUDE.md is authoritative where they differ.

# SpeechLens: kickoff

You are building SpeechLens with me: a contrastive speech analytics tool for the
Multimodal AI Hackathon 2026, Track C (Contrastive Speech Analytics & Temporal
Flaw Grounding). It compares a participant recording against a reference
recording of the same text, detects delivery flaws, and grounds each one in a
timestamp, a measurement, a named rule, and a plain explanation.

The architecture is already decided (below). Do not redesign it. If you find a
concrete reason a decision won't work, stop and tell me before deviating.

## Project facts
- Deadline: [DATE]. Team: [N people].
- Dataset speakers: [who records, how many]. English only: [yes/no].
- Repo licence: [MIT / GPL]. If MIT, praat-parselmouth (GPLv3) must not be a
  core dependency.
- Machine: Windows x64, RTX 3050 (4 GB VRAM), 8 GB RAM. Everything must run
  here. No paid APIs. No LLM in the scoring or detection path.

## Non-negotiable rules
1. Never fabricate measurements, scores, timestamps, accuracy numbers, dataset
   sizes, or chart data. If something can't be computed, return it as
   `unavailable` with a reason.
2. Every flaw carries: start/end seconds, category, severity, reference and
   participant measurements, deviation, rule_id, evidence, interpretation,
   recommendation. Keep observed evidence separate from inferred listener impact.
3. No fake buttons, placeholder charts, or TODO stubs for essential features.
   Ship a simpler working version and document the limitation instead.
4. Never claim forced alignment happened when the fallback ran. The alignment
   method is reported in every result and shown in the UI.
5. A deviation from the reference is not automatically a flaw. Rules score
   defined criteria only.
6. Don't say something works until you have run it. Show me the command and
   the output.

## Decided architecture
- Python 3.12 (librosa 1.0 requires it). Use uv, pin dependencies.
- Features: librosa for STFT, MFCC, RMS, and pYIN F0 with voiced flags.
  Configurable frame/hop sizes in config/features.yaml, units documented.
- Audio decode: PyAV + soundfile, so no system ffmpeg is needed. Keep the
  original file for playback; analyse a 16 kHz mono copy. Never
  peak-normalise the analysis copy before measuring energy.
- Alignment behind an adapter interface with two implementations:
  a) `energy_fallback`: VAD-based pauses and overall rate only, no word times.
  b) `ctc`: wav2vec2-base-960h emissions (transformers, Apache-2.0) plus our
     own Viterbi trellis in NumPy, returning per-word start/end and a score
     defined as mean frame log-probability along the path. Chunk long audio;
     cap uploads at 3 minutes.
  WhisperX is NOT a dependency (it pulls pyannote and pins torch 2.8).
- faster-whisper (base/small) is optional, only for drafting a transcript and
  flagging transcript mismatches.
- Comparison: match words by transcript index, never by time. Compare timing
  features (word durations, pauses, local rate) in word-indexed space with no
  warping. Use DTW only inside matched phrases for pitch/energy contour shape.
  Pitch in semitones relative to each speaker's median; energy in dB relative
  to each recording's median speech level. Skip pitch rules when too few
  frames are voiced.
- Fillers: forced alignment to a clean transcript cannot see them. Detect
  candidates as voiced stretches between aligned words that the transcript
  doesn't explain, and label them lower-reliability.
- Rules and rubric: versioned YAML in backend/config, pure deterministic
  functions, thresholds and weights never hardcoded in application code.
- Backend: one FastAPI process, Pydantic schemas, files on disk, SQLite only
  for analysis metadata.
- Frontend: React + Vite + TypeScript + Tailwind, WaveSurfer.js for the
  waveform and regions, Recharts for feature plots. Warm off-white background,
  charcoal/navy text, muted coral for flaws, restrained green/blue for
  comparisons. No gradients or glassmorphism.

## Layout
speechlens/
  backend/app/                 routes, schemas, job status
  backend/speechlens/{audio,features,alignment,compare,flaws,rubric,dataset}/
  backend/config/              rubric_v1.yaml, rules_v1.yaml, features.yaml
  backend/tests/
  frontend/
  dataset/                     manifest.jsonl, schema.json, annotations/, audio/
  scripts/                     build_dataset, evaluate, stress_test
  docs/                        architecture, dataset, rubric, evaluation, technical_report

## API
POST /api/recordings              upload; returns id, duration, sample rate, warnings
POST /api/analyses                participant id, reference id, transcript, rubric version -> job id
GET  /api/analyses/{id}           real stage status, then full result
GET  /api/recordings/{id}/audio   original audio
GET  /api/dataset/pairs, /api/dataset/pairs/{pair_id}
GET  /api/rubric                  active rules, thresholds, weights

## Dataset
One JSONL record per recording: recording_id, speech_id, pair_id,
reference_recording_id, speaker_id, transcript, condition (ideal | mild_flaw |
moderate_flaw | severe_flaw), flaw_categories, origin (human_recorded |
synthetic_transform), transform {type, params, source_recording_id},
audio_path, word_alignments_path, flaw_segments [{category, start_seconds,
end_seconds, severity}], annotation_method, recording_metadata, provenance
{recorded_by, licence, consent}, split.
- Passages are public-domain text recorded by our team. Do not download or
  redistribute third-party speech audio.
- Splits are assigned per speech_id so a passage never crosses splits.
- Synthetic transforms (inserted silence, time-stretched spans, gain changes,
  noise) record exact ground-truth boundaries and are always labelled and
  evaluated separately from human recordings.

## How to work
- First action: write CLAUDE.md capturing the rules, architecture, and commands
  above so they persist across sessions. Then `git init` and commit.
- Work in small commits. Write tests alongside each module, not afterwards.
- Before adding any dependency, check it installs on Windows with Python 3.12
  and tell me its size and licence.
- Ask me only what you can't determine by inspecting the repo or running code.
- At the end of each phase report: what works (with the commands you ran),
  files changed, how to run and test, what is incomplete, and the next step.
  Then wait for my go-ahead.

## Phase 1 (do this now): working vertical slice
1. Scaffold backend and frontend; confirm both start.
2. Audio decode, validation (format, duration, size, channels), 16 kHz copy.
3. RMS, VAD/pause detection, pYIN F0, overall speaking rate from transcript
   word count and speech time.
4. `energy_fallback` aligner behind the adapter interface.
5. Three rules with YAML thresholds: long pause, overall rate too fast/slow
   relative to the reference, flat pitch (semitone IQR). Rubric v1 with
   pacing, pause control, and vocal variation.
6. API endpoints for upload, analysis, status, audio.
7. Results page: both players, waveform with clickable flaw regions that seek
   playback, flaw list with measurements, rubric breakdown showing its inputs,
   and a visible "alignment: energy fallback" label.
8. scripts/build_dataset.py that takes one clean recording and produces
   synthetic pause-insertion variants with ground-truth flaw_segments.
9. Tests: pause detection on synthesised signals with known silences, F0 on a
   known-frequency tone, rubric determinism, flaw boundary accuracy against the
   synthetic ground truth, identical-recordings-produce-no-flaws, corrupted and
   empty audio handling.
10. README with exact Windows run instructions.

Later phases, for context only, do not start them:
2) CTC aligner, word-indexed comparison, local rate rules, contour DTW, fillers.
3) Recorded dataset, annotations, evaluation (F1, temporal IoU, repeatability).
4) Synchronised plots, transcript highlighting, dataset explorer.
5) Requirement audit, six-page technical report, demo checklist.

Start by inspecting the folder, then write CLAUDE.md and show it to me before
scaffolding.
