"""CTC forced alignment: wav2vec2-base-960h emissions + NumPy Viterbi trellis.

Time resolution: one frame per 320 samples at 16 kHz = 20 ms.
"""
from __future__ import annotations

import string

import numpy as np
from numpy.typing import NDArray

from speechlens.alignment.types import AlignmentResult, WordTiming
from speechlens.features.extract import VADResult

# ── wav2vec2-base-960h vocabulary ────────────────────────────────────────
VOCAB: dict[str, int] = {
    "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4,
    "E": 5, "T": 6, "A": 7, "O": 8, "N": 9, "I": 10,
    "H": 11, "S": 12, "R": 13, "D": 14, "L": 15, "U": 16,
    "M": 17, "W": 18, "C": 19, "F": 20, "G": 21, "Y": 22,
    "P": 23, "B": 24, "V": 25, "K": 26, "'": 27, "X": 28,
    "J": 29, "Q": 30, "Z": 31,
}
BLANK_IDX = 0
SEP_IDX = 4
VOCAB_SIZE = 32
FRAME_STRIDE = 320  # samples per emission frame

_REMOVABLE = set(string.punctuation) | set("‘’“”–—…")


# ── transcript normalisation ─────────────────────────────────────────────

def normalize_transcript(
    transcript: str,
    vocab: dict[str, int] | None = None,
) -> tuple[list[int], list[str]]:
    """Normalise transcript to wav2vec2-base-960h token indices.

    Uppercases, strips punctuation, replaces spaces with ``|``.
    Raises ``ValueError`` naming every word that contains a character
    outside the model vocabulary (such as digits).

    Parameters
    ----------
    transcript : str
        Raw transcript text.
    vocab : dict, optional
        Character-to-index mapping. Defaults to the module-level ``VOCAB``.

    Returns
    -------
    tokens : list[int]
        Token indices including ``|`` separators between words.
    words : list[str]
        Cleaned uppercase words (no separators).
    """
    if vocab is None:
        vocab = VOCAB

    text = transcript.strip()
    if not text:
        raise ValueError("Transcript is empty")

    raw_words = text.split()
    bad_words: list[str] = []
    clean_words: list[str] = []

    for raw in raw_words:
        letters: list[str] = []
        has_invalid = False
        for ch in raw:
            upper = ch.upper()
            if upper.isascii() and upper.isalpha():
                letters.append(upper)
            elif ch in _REMOVABLE or ch.isspace():
                pass
            else:
                has_invalid = True
        if has_invalid:
            bad_words.append(raw)
        elif letters:
            clean_words.append("".join(letters))

    if bad_words:
        raise ValueError(
            "Words with characters outside the wav2vec2 vocabulary: "
            + ", ".join(repr(w) for w in bad_words)
        )
    if not clean_words:
        raise ValueError("Transcript is empty after normalization")

    tokens: list[int] = []
    for i, w in enumerate(clean_words):
        if i > 0:
            tokens.append(vocab["|"])
        for ch in w:
            tokens.append(vocab[ch])
    return tokens, clean_words


# ── Viterbi trellis (pure NumPy) ─────────────────────────────────────────

def _build_expanded(tokens: list[int]) -> NDArray[np.int32]:
    """Interleave blanks: ``[blank, t0, blank, t1, ..., tN, blank]``."""
    S = len(tokens)
    exp = np.zeros(2 * S + 1, dtype=np.int32)
    for i in range(S):
        exp[2 * i + 1] = tokens[i]
    return exp


def viterbi_trellis(
    log_probs: NDArray[np.float32],
    tokens: list[int],
) -> tuple[NDArray[np.float32], NDArray[np.int8], NDArray[np.int32]]:
    """Viterbi best-path trellis over CTC expanded token sequence.

    Parameters
    ----------
    log_probs : ndarray, shape (T, C)
        Per-frame log-softmax output.
    tokens : list[int]
        Target token indices **without** interleaved blanks.

    Returns
    -------
    trellis : ndarray (T, L)
        Best log-prob at each (frame, expanded position).
    backptrs : ndarray (T, L)
        Transition taken: 0 = stay, 1 = from j-1, 2 = from j-2.
    expanded : ndarray (L,)
        Expanded token sequence with blanks interleaved.
    """
    T = log_probs.shape[0]
    expanded = _build_expanded(tokens)
    L = len(expanded)

    min_frames = (L + 1) // 2
    if T < min_frames:
        raise ValueError(
            f"Audio too short: {T} frames for {len(tokens)} tokens "
            f"(need >= {min_frames} frames)"
        )

    NEG_INF = np.float32(-1e9)
    trellis = np.full((T, L), NEG_INF, dtype=np.float32)
    backptrs = np.zeros((T, L), dtype=np.int8)

    trellis[0, 0] = log_probs[0, expanded[0]]
    if L > 1:
        trellis[0, 1] = log_probs[0, expanded[1]]

    skip_ok = np.zeros(L, dtype=bool)
    if L > 2:
        skip_ok[2:] = expanded[2:] != expanded[:-2]

    for t in range(1, T):
        emit = log_probs[t, expanded]

        stay = trellis[t - 1].copy()

        step1 = np.full(L, NEG_INF, dtype=np.float32)
        step1[1:] = trellis[t - 1, :-1]

        step2 = np.full(L, NEG_INF, dtype=np.float32)
        if L > 2:
            step2[2:] = np.where(skip_ok[2:], trellis[t - 1, :-2], NEG_INF)

        cands = np.stack([stay, step1, step2])
        best = cands.argmax(axis=0).astype(np.int8)
        trellis[t] = cands[best, np.arange(L)] + emit
        backptrs[t] = best

    return trellis, backptrs, expanded


def backtrack(
    trellis: NDArray[np.float32],
    backptrs: NDArray[np.int8],
    expanded: NDArray[np.int32],
    log_probs: NDArray[np.float32],
) -> list[tuple[int, int, float]]:
    """Walk backward through the trellis.

    Returns one ``(frame, expanded_position, emission_log_prob)`` per frame,
    ordered by ascending frame index.
    """
    T, L = trellis.shape

    j = L - 1
    if L >= 2 and trellis[T - 1, L - 2] > trellis[T - 1, L - 1]:
        j = L - 2

    path: list[tuple[int, int, float]] = []
    for t in range(T - 1, -1, -1):
        path.append((t, j, float(log_probs[t, expanded[j]])))
        if t > 0:
            j -= int(backptrs[t, j])

    path.reverse()
    return path


def merge_words(
    path: list[tuple[int, int, float]],
    tokens: list[int],
    words: list[str],
    frame_duration: float,
) -> list[WordTiming]:
    """Collapse a frame-level CTC path into one ``WordTiming`` per word.

    Score = mean emission log-prob of non-blank character frames in the word.
    """
    tok_to_word: dict[int, int] = {}
    wi = 0
    for ti, tok in enumerate(tokens):
        if tok == SEP_IDX:
            wi += 1
        else:
            tok_to_word[ti] = wi

    word_frames: dict[int, list[tuple[int, float]]] = {
        i: [] for i in range(len(words))
    }
    for frame_idx, exp_pos, lp in path:
        if exp_pos % 2 == 1:
            ti = exp_pos // 2
            if ti in tok_to_word:
                word_frames[tok_to_word[ti]].append((frame_idx, lp))

    result: list[WordTiming] = []
    for wi_idx, word in enumerate(words):
        frames = word_frames[wi_idx]
        if not frames:
            continue
        first = frames[0][0]
        last = frames[-1][0]
        score = float(np.mean([lp for _, lp in frames]))
        result.append(WordTiming(
            word=word,
            start=round(first * frame_duration, 4),
            end=round((last + 1) * frame_duration, 4),
            score=round(score, 4),
        ))
    return result


# ── CTCAligner (requires torch + transformers) ───────────────────────────

class CTCAligner:
    """CTC forced aligner using wav2vec2-base-960h.

    Requires the ``align`` optional extra (``uv sync --extra align``).
    """

    CHUNK_SAMPLES = 30 * 16_000

    def __init__(self, device: str | None = None) -> None:
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        name = "facebook/wav2vec2-base-960h"
        self._processor = Wav2Vec2Processor.from_pretrained(name)
        self._model = Wav2Vec2ForCTC.from_pretrained(name).to(self._device)
        self._model.eval()

        self._vocab: dict[str, int] = self._processor.tokenizer.get_vocab()

        for char, idx in VOCAB.items():
            if self._vocab.get(char) != idx:
                raise RuntimeError(
                    f"Vocab mismatch: expected {char!r}={idx}, "
                    f"tokenizer has {self._vocab.get(char)}"
                )

    def get_emissions(
        self, samples: NDArray[np.float32], sample_rate: int = 16_000,
    ) -> NDArray[np.float32]:
        """Return ``(T, 32)`` log-softmax emissions, chunking long audio."""
        import torch

        assert sample_rate == 16_000
        n = len(samples)
        chunks = (
            [samples]
            if n <= self.CHUNK_SAMPLES
            else [samples[i : i + self.CHUNK_SAMPLES]
                  for i in range(0, n, self.CHUNK_SAMPLES)]
        )

        parts: list[NDArray[np.float32]] = []
        for chunk in chunks:
            iv = self._processor(
                chunk, sampling_rate=16_000, return_tensors="pt",
            ).input_values.to(self._device)
            with torch.no_grad():
                logits = self._model(iv).logits
            lp = torch.log_softmax(logits, dim=-1)[0].cpu().numpy()
            parts.append(lp.astype(np.float32))

        return np.concatenate(parts, axis=0)

    def align(
        self,
        samples: NDArray[np.float32],
        sample_rate: int,
        transcript: str,
        vad: VADResult,
    ) -> AlignmentResult:
        tokens, words = normalize_transcript(transcript, vocab=self._vocab)
        log_probs = self.get_emissions(samples, sample_rate)
        fd = FRAME_STRIDE / sample_rate

        tr, bp, expanded = viterbi_trellis(log_probs, tokens)
        path = backtrack(tr, bp, expanded, log_probs)
        word_timings = merge_words(path, tokens, words, fd)

        return AlignmentResult(
            method="ctc",
            words=word_timings,
            pauses=list(vad.pauses),
            unavailable=[],
        )
