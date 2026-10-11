"""Run rules and rubric comparison between a reference and participant.

Usage:
    uv run python scripts/run_comparison.py <reference_wav> <participant_wav> <transcript_path>

Prints the full result as JSON: every flaw, every rubric dimension, and the
unavailable list.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from speechlens.audio.decode import decode_audio
from speechlens.features.extract import extract_features
from speechlens.flaws.rules import detect_all_flaws
from speechlens.rubric.rubric import score_rubric


def main() -> None:
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <reference_wav> <participant_wav> <transcript_path>")
        sys.exit(1)

    ref_path = Path(sys.argv[1])
    part_path = Path(sys.argv[2])
    transcript = Path(sys.argv[3]).read_text(encoding="utf-8").strip()

    ref_decoded = decode_audio(ref_path)
    part_decoded = decode_audio(part_path)

    ref_dur = len(ref_decoded.samples) / ref_decoded.sample_rate
    part_dur = len(part_decoded.samples) / part_decoded.sample_rate

    ref_feat = extract_features(ref_decoded.samples, ref_decoded.sample_rate, transcript, ref_dur)
    part_feat = extract_features(part_decoded.samples, part_decoded.sample_rate, transcript, part_dur)

    flaws, unavailable = detect_all_flaws(part_feat, ref_feat)
    rubric = score_rubric(part_feat, ref_feat, flaws, unavailable)

    result = {
        "reference": str(ref_path.name),
        "participant": str(part_path.name),
        "flaws": [dataclasses.asdict(f) for f in flaws],
        "unavailable": unavailable,
        "rubric": {
            "overall_score": rubric.overall_score,
            "version": rubric.version,
            "dimensions": [dataclasses.asdict(d) for d in rubric.dimensions],
        },
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
