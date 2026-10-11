"""CTC trellis tests on hand-built emission matrices (no torch required)."""
from __future__ import annotations

import numpy as np
import pytest

from speechlens.alignment.ctc import (
    BLANK_IDX,
    SEP_IDX,
    VOCAB,
    VOCAB_SIZE,
    _build_expanded,
    backtrack,
    merge_words,
    normalize_transcript,
    viterbi_trellis,
)


# ── helpers ──────────────────────────────────────────────────────────────

def _peaked(T: int, schedule: list[tuple[int, int]]) -> np.ndarray:
    """Emissions strongly peaked at scheduled (frame, token) pairs."""
    lp = np.full((T, VOCAB_SIZE), -10.0, dtype=np.float32)
    for frame, tok in schedule:
        lp[frame, tok] = -0.1
    return lp


# ── normalize_transcript ─────────────────────────────────────────────────

class TestNormalizeTranscript:
    def test_simple(self):
        tokens, words = normalize_transcript("Hello world")
        assert words == ["HELLO", "WORLD"]
        expected = [
            VOCAB["H"], VOCAB["E"], VOCAB["L"], VOCAB["L"], VOCAB["O"],
            SEP_IDX,
            VOCAB["W"], VOCAB["O"], VOCAB["R"], VOCAB["L"], VOCAB["D"],
        ]
        assert tokens == expected

    def test_strips_punctuation(self):
        _, words = normalize_transcript("Hello, world!")
        assert words == ["HELLO", "WORLD"]

    def test_strips_curly_quotes(self):
        _, words = normalize_transcript("“Hello”")
        assert words == ["HELLO"]

    def test_digits_raise(self):
        with pytest.raises(ValueError, match="3rd"):
            normalize_transcript("The 3rd time")

    def test_multiple_bad_words(self):
        with pytest.raises(ValueError) as exc_info:
            normalize_transcript("1st and 2nd")
        msg = str(exc_info.value)
        assert "1st" in msg
        assert "2nd" in msg

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            normalize_transcript("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="empty"):
            normalize_transcript("   ")


# ── _build_expanded ──────────────────────────────────────────────────────

class TestBuildExpanded:
    def test_two_tokens(self):
        exp = _build_expanded([5, 6])
        np.testing.assert_array_equal(exp, [0, 5, 0, 6, 0])

    def test_single_token(self):
        exp = _build_expanded([7])
        np.testing.assert_array_equal(exp, [0, 7, 0])


# ── viterbi_trellis + backtrack ──────────────────────────────────────────

class TestViterbiTrellis:
    def test_two_words(self):
        """Target 'A B': path through blank-A-blank-|-blank-B-blank."""
        A, B = VOCAB["A"], VOCAB["B"]
        tokens = [A, SEP_IDX, B]

        lp = _peaked(8, [
            (0, BLANK_IDX), (1, A), (2, A),
            (3, BLANK_IDX), (4, SEP_IDX),
            (5, BLANK_IDX), (6, B), (7, BLANK_IDX),
        ])
        tr, bp, expanded = viterbi_trellis(lp, tokens)
        path = backtrack(tr, bp, expanded, lp)

        positions = [p[1] for p in path]
        assert positions == [0, 1, 1, 2, 3, 4, 5, 6]

        wt = merge_words(path, tokens, ["A", "B"], 0.02)
        assert len(wt) == 2
        assert wt[0].word == "A"
        assert wt[0].start == pytest.approx(0.02)
        assert wt[0].end == pytest.approx(0.06)
        assert wt[1].word == "B"
        assert wt[1].start == pytest.approx(0.12)
        assert wt[1].end == pytest.approx(0.14)

    def test_repeated_char_requires_blank(self):
        """Target 'ALL': blank required between the two Ls."""
        A, L = VOCAB["A"], VOCAB["L"]
        tokens = [A, L, L]

        lp = _peaked(7, [
            (0, BLANK_IDX), (1, A), (2, L),
            (3, BLANK_IDX), (4, L),
            (5, BLANK_IDX), (6, BLANK_IDX),
        ])
        tr, bp, expanded = viterbi_trellis(lp, tokens)
        path = backtrack(tr, bp, expanded, lp)

        positions = [p[1] for p in path]
        assert positions == [0, 1, 3, 4, 5, 6, 6]

        wt = merge_words(path, tokens, ["ALL"], 0.02)
        assert len(wt) == 1
        assert wt[0].word == "ALL"
        assert wt[0].start == pytest.approx(0.02)
        assert wt[0].end == pytest.approx(0.10)

    def test_score_is_mean_char_log_prob(self):
        """Score = mean log-prob of non-blank character frames."""
        A = VOCAB["A"]
        lp = np.full((4, VOCAB_SIZE), -10.0, dtype=np.float32)
        lp[0, BLANK_IDX] = -0.5
        lp[1, A] = -0.2
        lp[2, A] = -0.4
        lp[3, BLANK_IDX] = -0.7

        tr, bp, expanded = viterbi_trellis(lp, [A])
        path = backtrack(tr, bp, expanded, lp)
        wt = merge_words(path, [A], ["A"], 0.02)

        assert len(wt) == 1
        assert wt[0].score == pytest.approx(-0.3, abs=1e-4)

    def test_minimum_frames(self):
        """Two frames suffice for one token (expanded length 3)."""
        A = VOCAB["A"]
        lp = np.full((2, VOCAB_SIZE), -10.0, dtype=np.float32)
        lp[0, A] = -0.1
        lp[1, BLANK_IDX] = -0.1

        tr, bp, expanded = viterbi_trellis(lp, [A])
        path = backtrack(tr, bp, expanded, lp)

        positions = [p[1] for p in path]
        assert positions == [1, 2]

        wt = merge_words(path, [A], ["A"], 0.02)
        assert len(wt) == 1
        assert wt[0].start == pytest.approx(0.0)
        assert wt[0].end == pytest.approx(0.02)

    def test_too_few_frames_raises(self):
        """Audio shorter than minimum raises ValueError."""
        A, B = VOCAB["A"], VOCAB["B"]
        lp = np.full((1, VOCAB_SIZE), -0.1, dtype=np.float32)
        with pytest.raises(ValueError, match="too short"):
            viterbi_trellis(lp, [A, SEP_IDX, B])

    def test_three_word_sentence(self):
        """Target 'I AM OK': three single-char words."""
        I, A, M, O, K = VOCAB["I"], VOCAB["A"], VOCAB["M"], VOCAB["O"], VOCAB["K"]
        tokens = [I, SEP_IDX, A, M, SEP_IDX, O, K]

        lp = _peaked(10, [
            (0, BLANK_IDX), (1, I),
            (2, SEP_IDX), (3, A), (4, M),
            (5, BLANK_IDX), (6, SEP_IDX),
            (7, O), (8, K), (9, BLANK_IDX),
        ])
        tr, bp, expanded = viterbi_trellis(lp, tokens)
        path = backtrack(tr, bp, expanded, lp)
        wt = merge_words(path, tokens, ["I", "AM", "OK"], 0.02)

        assert len(wt) == 3
        assert [w.word for w in wt] == ["I", "AM", "OK"]
        for w in wt:
            assert w.end > w.start

    def test_path_log_probs_sum(self):
        """Total path log-prob equals sum over frame emissions."""
        A = VOCAB["A"]
        lp = np.full((3, VOCAB_SIZE), -10.0, dtype=np.float32)
        lp[0, BLANK_IDX] = -1.0
        lp[1, A] = -2.0
        lp[2, BLANK_IDX] = -3.0

        tr, bp, expanded = viterbi_trellis(lp, [A])
        path = backtrack(tr, bp, expanded, lp)

        path_sum = sum(lp_val for _, _, lp_val in path)
        assert path_sum == pytest.approx(-6.0)
        assert tr[-1, path[-1][1]] == pytest.approx(-6.0)
