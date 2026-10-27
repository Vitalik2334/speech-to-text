"""
asr_metrics.py — оцінка тексту розпізнавання через jiwer.

  CER (по символах) — основна метрика: для флективної української
      "зробив"/"зробила" — дрібна помилка, а не повністю чуже слово.
  WER (по словах)   — додаткова: скільки слів доведеться правити руками.

Загальні цифри рахуються ПО СУМІ помилок і довжин, а не як середнє
відсотків: коротка фраза з 2 слів не важить стільки ж, скільки монолог.

Окремо — пунктуація на внутрішніх межах речень (крапка/?/! після слова).
Кінець усього тексту не рахується: там крапку ставить і так будь-хто.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import jiwer

# --------------------------------------------------------------------------
# Нормалізація
# --------------------------------------------------------------------------

_APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'", "‘": "'", "`": "'", "´": "'"})
_NOT_WORD = re.compile(r"[^\w\s']", re.UNICODE)
_EDGE_APOSTROPHE = re.compile(r"(?<!\w)'|'(?!\w)")
_SPACES = re.compile(r"\s+")
_SENTENCE_END = (".", "?", "!", "…")
_CLOSING = "\"'»”)]"


def normalize(text: str) -> str:
    """Нижній регістр, один вид апострофа, без пунктуації, один пробіл."""
    text = unicodedata.normalize("NFC", text or "").lower().translate(_APOSTROPHES)
    text = _NOT_WORD.sub(" ", text).replace("_", " ")
    text = _EDGE_APOSTROPHE.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


# --------------------------------------------------------------------------
# Оцінка одного кліпу
# --------------------------------------------------------------------------

GRADES: tuple[tuple[float, str], ...] = (
    (0.03, "відмінно"),
    (0.08, "добре"),
    (0.15, "прийнятно"),
)


def grade(cer: float) -> str:
    for limit, name in GRADES:
        if cer <= limit:
            return name
    return "погано"


@dataclass
class PunctStats:
    tp: int = 0     # межа і в еталоні, і в результаті
    fp: int = 0     # зайва межа в результаті
    fn: int = 0     # пропущена межа

    def __add__(self, other: "PunctStats") -> "PunctStats":
        return PunctStats(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    @property
    def f1(self) -> float | None:
        denom = 2 * self.tp + self.fp + self.fn
        return None if denom == 0 else 2 * self.tp / denom


@dataclass
class ClipScore:
    reference: str
    hypothesis: str
    word_errors: int
    word_total: int
    char_errors: int
    char_total: int
    diff: str
    punct: PunctStats

    @property
    def cer(self) -> float:
        return self.char_errors / self.char_total if self.char_total else 0.0

    @property
    def wer(self) -> float:
        return self.word_errors / self.word_total if self.word_total else 0.0

    @property
    def grade(self) -> str:
        return grade(self.cer)


def _counts(output) -> tuple[int, int]:
    errors = output.substitutions + output.deletions + output.insertions
    total = output.substitutions + output.deletions + output.hits
    return errors, total


def _word_diff(output) -> str:
    """Компактний диф: [ref→hyp] заміна, [-ref] пропуск, [+hyp] вставка."""
    ref, hyp = output.references[0], output.hypotheses[0]
    parts: list[str] = []
    for chunk in output.alignments[0]:
        r = " ".join(ref[chunk.ref_start_idx:chunk.ref_end_idx])
        h = " ".join(hyp[chunk.hyp_start_idx:chunk.hyp_end_idx])
        if chunk.type == "equal":
            parts.append(r)
        elif chunk.type == "substitute":
            parts.append(f"[{r}→{h}]")
        elif chunk.type == "delete":
            parts.append(f"[-{r}]")
        else:
            parts.append(f"[+{h}]")
    return " ".join(parts)


def score_clip(reference: str, hypothesis: str) -> ClipScore:
    """Оцінка кліпу з НЕПОРОЖНІМ еталоном (порожні — негативні кліпи,
    їх перевіряють окремо: результат має бути порожнім)."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        raise ValueError("score_clip: порожній еталон — це негативний кліп")
    if hyp:
        words = jiwer.process_words(ref, hyp)
        chars = jiwer.process_characters(ref, hyp)
        word_errors, word_total = _counts(words)
        char_errors, char_total = _counts(chars)
        diff = _word_diff(words)
    else:
        # jiwer не приймає порожній текст — усе сказане пропущено.
        word_total, char_total = len(ref.split()), len(ref)
        word_errors, char_errors = word_total, char_total
        diff = f"[-{ref}]"
    return ClipScore(reference, hypothesis, word_errors, word_total, char_errors, char_total,
                     diff, punctuation_stats(reference, hypothesis))


# --------------------------------------------------------------------------
# Пунктуація на межах речень
# --------------------------------------------------------------------------

def _words_with_boundaries(text: str) -> tuple[list[str], list[bool]]:
    words: list[str] = []
    ends: list[bool] = []
    for token in (text or "").split():
        is_end = token.rstrip(_CLOSING).endswith(_SENTENCE_END)
        word = normalize(token)
        if word:
            for w in word.split():
                words.append(w)
                ends.append(False)
            ends[-1] = is_end
        elif is_end and ends:          # окремий знак (" ." / " —")
            ends[-1] = True
    return words, ends


def punctuation_stats(reference: str, hypothesis: str) -> PunctStats:
    ref_words, ref_ends = _words_with_boundaries(reference)
    hyp_words, hyp_ends = _words_with_boundaries(hypothesis)
    stats = PunctStats()
    if not ref_words or not hyp_words:
        return stats
    out = jiwer.process_words(" ".join(ref_words), " ".join(hyp_words))
    last_ref = len(ref_words) - 1
    for chunk in out.alignments[0]:
        if chunk.type not in ("equal", "substitute"):
            continue
        for k in range(chunk.ref_end_idx - chunk.ref_start_idx):
            r, h = chunk.ref_start_idx + k, chunk.hyp_start_idx + k
            if r == last_ref:
                continue
            if ref_ends[r] and hyp_ends[h]:
                stats.tp += 1
            elif hyp_ends[h]:
                stats.fp += 1
            elif ref_ends[r]:
                stats.fn += 1
    return stats


# --------------------------------------------------------------------------
# Підсумки
# --------------------------------------------------------------------------

@dataclass
class Totals:
    clips: int = 0
    word_errors: int = 0
    word_total: int = 0
    char_errors: int = 0
    char_total: int = 0
    punct: PunctStats = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.punct is None:
            self.punct = PunctStats()

    @property
    def cer(self) -> float | None:
        return self.char_errors / self.char_total if self.char_total else None

    @property
    def wer(self) -> float | None:
        return self.word_errors / self.word_total if self.word_total else None


def aggregate(scores: list[ClipScore]) -> Totals:
    t = Totals()
    for s in scores:
        t.clips += 1
        t.word_errors += s.word_errors
        t.word_total += s.word_total
        t.char_errors += s.char_errors
        t.char_total += s.char_total
        t.punct = t.punct + s.punct
    return t
