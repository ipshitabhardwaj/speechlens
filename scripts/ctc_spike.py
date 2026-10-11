"""CTC aligner spike: align a WAV against a transcript, print word timings."""
from __future__ import annotations

import argparse
import sys
import time
import tracemalloc

import numpy as np
import soundfile as sf


def main() -> None:
    ap = argparse.ArgumentParser(description="CTC forced-alignment spike")
    ap.add_argument("wav", help="16 kHz mono WAV (resampled automatically if needed)")
    ap.add_argument("transcript", help="plain-text transcript file")
    ap.add_argument("--device", default=None, help="cuda | cpu (auto-detects)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="tile audio+transcript N times for stress testing")
    ap.add_argument("--vad-check", action="store_true",
                    help="compare CTC word boundaries against VAD pause boundaries")
    args = ap.parse_args()

    # ── load audio ───────────────────────────────────────────────────────
    samples, sr = sf.read(args.wav, dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sr != 16_000:
        import soxr
        samples = soxr.resample(samples, sr, 16_000).astype(np.float32)
        sr = 16_000

    transcript = open(args.transcript, encoding="utf-8").read().strip()

    if args.repeat > 1:
        samples = np.tile(samples, args.repeat)
        transcript = " ".join([transcript] * args.repeat)

    print(f"Audio: {len(samples) / sr:.1f} s, {len(samples)} samples")
    print(f"Transcript: {len(transcript.split())} words")

    # ── align ────────────────────────────────────────────────────────────
    from speechlens.alignment.ctc import (
        CTCAligner,
        FRAME_STRIDE,
        backtrack,
        merge_words,
        normalize_transcript,
        viterbi_trellis,
    )

    tracemalloc.start()
    t0 = time.perf_counter()

    aligner = CTCAligner(device=args.device)
    t_load = time.perf_counter() - t0

    tokens, words = normalize_transcript(transcript)

    t1 = time.perf_counter()
    log_probs = aligner.get_emissions(samples, sr)
    t_emit = time.perf_counter() - t1

    t2 = time.perf_counter()
    tr, bp, expanded = viterbi_trellis(log_probs, tokens)
    path = backtrack(tr, bp, expanded, log_probs)
    fd = FRAME_STRIDE / sr
    wt = merge_words(path, tokens, words, fd)
    t_trellis = time.perf_counter() - t2

    total = time.perf_counter() - t0
    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # ── report ───────────────────────────────────────────────────────────
    print(f"\nEmissions: {log_probs.shape[0]} frames x {log_probs.shape[1]} classes")
    print(f"Expanded target length: {len(expanded)}")
    print(f"Device: {aligner._device}")
    print(f"Time: model load {t_load:.2f} s, emissions {t_emit:.2f} s, "
          f"trellis {t_trellis:.2f} s, total {total:.2f} s")
    print(f"Peak RAM (tracemalloc): {peak_ram / 1024**2:.1f} MB")

    import torch
    if aligner._device.type == "cuda":
        vram = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Peak VRAM: {vram:.1f} MB")

    # ── first 20 words ───────────────────────────────────────────────────
    print(f"\n{'word':<20s} {'start':>8s} {'end':>8s} {'score':>8s}")
    print("-" * 48)
    for w in wt[:20]:
        print(f"{w.word:<20s} {w.start:8.3f} {w.end:8.3f} {w.score:8.4f}")
    print(f"\nTotal words aligned: {len(wt)}")

    # ── checks ───────────────────────────────────────────────────────────
    print("\n--- Checks ---")
    ok = True

    if len(wt) != len(words):
        print(f"FAIL: expected {len(words)} words, got {len(wt)}")
        ok = False
    else:
        print(f"OK: all {len(words)} words present exactly once")

    for i in range(1, len(wt)):
        if wt[i].start < wt[i - 1].end:
            print(f"FAIL: overlap at word {i}: {wt[i - 1].word} ends "
                  f"{wt[i - 1].end:.4f}, {wt[i].word} starts {wt[i].start:.4f}")
            ok = False
            break
    else:
        print("OK: times monotonic and non-overlapping")

    for w in wt:
        if w.end <= w.start:
            print(f"FAIL: zero-length word {w.word} at {w.start}")
            ok = False
            break
    else:
        print("OK: no zero-length words")

    if not ok:
        sys.exit(1)

    # ── optional VAD accuracy check ──────────────────────────────────────
    if args.vad_check:
        print("\n--- VAD accuracy check ---")
        from speechlens.features.extract import extract_features

        feat = extract_features(samples, sr)
        long_pauses = [p for p in feat.vad.pauses if (p.end - p.start) > 1.0]
        print(f"Long pauses from VAD (> 1 s): {len(long_pauses)}")

        for pi, pause in enumerate(long_pauses):
            dur = pause.end - pause.start
            before = None
            for w in wt:
                if w.end <= pause.start + 0.1:
                    before = w
            after = None
            for w in wt:
                if w.start >= pause.end - 0.1:
                    after = w
                    break

            print(f"\nPause {pi + 1}: VAD [{pause.start:.3f}, {pause.end:.3f}] s  "
                  f"(duration {dur:.3f} s)")
            if before:
                err = (before.end - pause.start) * 1000
                print(f"  Word before: '{before.word}' ends {before.end:.3f} s  "
                      f"-> error {err:+.0f} ms")
            if after:
                err = (after.start - pause.end) * 1000
                print(f"  Word after:  '{after.word}' starts {after.start:.3f} s  "
                      f"-> error {err:+.0f} ms")


if __name__ == "__main__":
    main()
