"""
report.py — підсумки прогону, базова лінія (baseline.json) і текстовий звіт.

baseline.json — "знімок того, як добре працювало востаннє, коли вас усе
влаштовувало". Тести порівнюють кожен новий прогін з ним. Оновлюється
лише явно: pytest -m integration --update-baseline.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from Tests.recognition.asr_metrics import ClipScore, Totals, aggregate, normalize, score_clip

REPORT_KEY = pytest.StashKey[str]()
BASELINE_FILE = "baseline.json"
REPORTS_DIR = "reports"


def ref_hash(reference: str) -> str:
    return hashlib.sha1(normalize(reference).encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------
# Підсумки прогону
# --------------------------------------------------------------------------

@dataclass
class Evaluation:
    config: dict
    runs: list                                   # list[ClipRun]
    scores: dict[str, ClipScore] = field(default_factory=dict)         # потоковий цикл
    single_scores: dict[str, ClipScore] = field(default_factory=dict)  # один прохід
    created_at: str = field(default_factory=lambda: dt.datetime.now().astimezone().isoformat(timespec="seconds"))

    def __post_init__(self) -> None:
        for run in self.runs:
            if run.clip.expect_empty:
                continue
            self.scores[run.clip.id] = score_clip(run.clip.reference, run.recognized)
            if run.single_pass is not None:
                self.single_scores[run.clip.id] = score_clip(run.clip.reference, run.single_pass)

    @property
    def positive_runs(self) -> list:
        return [r for r in self.runs if not r.clip.expect_empty]

    @property
    def negative_runs(self) -> list:
        return [r for r in self.runs if r.clip.expect_empty]

    @property
    def total(self) -> Totals:
        return aggregate(list(self.scores.values()))

    @property
    def single_total(self) -> Totals:
        return aggregate(list(self.single_scores.values()))

    def by_tag(self) -> dict[str, Totals]:
        groups: dict[str, list[ClipScore]] = {}
        for run in self.positive_runs:
            for tag in run.clip.tags:
                groups.setdefault(tag, []).append(self.scores[run.clip.id])
        return {tag: aggregate(scores) for tag, scores in sorted(groups.items())}

    @property
    def streaming_gap_cer(self) -> float | None:
        a, b = self.total.cer, self.single_total.cer
        return None if a is None or b is None else a - b

    def failed_negatives(self) -> list:
        return [r for r in self.negative_runs if normalize(r.recognized)]

    # ---- baseline -----------------------------------------------------------

    def to_baseline(self) -> dict:
        total = self.total
        return {
            "created_at": self.created_at,
            "model": self.config.get("ACTIVE_MODEL"),
            "quantization": self.config.get("QUANTIZATION"),
            "clips_count": len(self.runs),
            "total": {"cer": total.cer, "wer": total.wer},
            "by_tag": {t: {"cer": v.cer, "wer": v.wer} for t, v in self.by_tag().items()},
            "streaming_gap_cer": self.streaming_gap_cer,
            "punct_f1": total.punct.f1,
            "clips": {
                r.clip.id: {
                    "ref": ref_hash(r.clip.reference),
                    "cer": None if r.clip.expect_empty else self.scores[r.clip.id].cer,
                    "wer": None if r.clip.expect_empty else self.scores[r.clip.id].wer,
                }
                for r in self.runs
            },
            "config": self.config,
        }


# --------------------------------------------------------------------------
# Порівняння з baseline
# --------------------------------------------------------------------------

def load_baseline(dataset_dir: Path) -> dict | None:
    path = Path(dataset_dir) / BASELINE_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save_baseline(dataset_dir: Path, evaluation: Evaluation) -> Path:
    path = Path(dataset_dir) / BASELINE_FILE
    path.write_text(json.dumps(evaluation.to_baseline(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def baseline_mismatch(evaluation: Evaluation, baseline: dict) -> str | None:
    """Причина, чому порівнювати з baseline некоректно, або None."""
    cfg = evaluation.config
    if baseline.get("model") != cfg.get("ACTIVE_MODEL") or baseline.get("quantization") != cfg.get("QUANTIZATION"):
        return (f"baseline записано для {baseline.get('model')} / {baseline.get('quantization') or 'fp32'}, "
                f"а зараз {cfg.get('ACTIVE_MODEL')} / {cfg.get('QUANTIZATION') or 'fp32'}")
    now = {r.clip.id: ref_hash(r.clip.reference) for r in evaluation.runs}
    before = {cid: info.get("ref") for cid, info in baseline.get("clips", {}).items()}
    added = sorted(now.keys() - before.keys())
    removed = sorted(before.keys() - now.keys())
    edited = sorted(cid for cid in now.keys() & before.keys() if now[cid] != before[cid])
    if added or removed or edited:
        parts = []
        if added:
            parts.append(f"додано {', '.join(added)}")
        if removed:
            parts.append(f"прибрано {', '.join(removed)}")
        if edited:
            parts.append(f"змінено еталон {', '.join(edited)}")
        return "набір кліпів змінився (" + "; ".join(parts) + ")"
    return None


def config_changes(evaluation: Evaluation, baseline: dict | None) -> list[str]:
    if not baseline:
        return []
    old = baseline.get("config", {})
    keys = sorted(set(old) | set(evaluation.config))
    return [f"{k}: {old.get(k, '—')} → {evaluation.config.get(k, '—')}"
            for k in keys if old.get(k) != evaluation.config.get(k)]


# --------------------------------------------------------------------------
# Звіт
# --------------------------------------------------------------------------

def _pct(value: float | None) -> str:
    return "   —  " if value is None else f"{value * 100:5.1f}%"


def _delta(now: float | None, before: float | None) -> str:
    if now is None or before is None:
        return ""
    return f"{(now - before) * 100:+5.1f}"


def build_report(evaluation: Evaluation, baseline: dict | None, mismatch: str | None) -> str:
    cfg = evaluation.config
    total, single = evaluation.total, evaluation.single_total
    comparable = baseline if baseline and not mismatch else None
    base_clips = comparable.get("clips", {}) if comparable else {}

    lines = [
        f"Модель: {cfg.get('ACTIVE_MODEL')}   {cfg.get('QUANTIZATION') or 'fp32'}   "
        f"кліпів: {len(evaluation.runs)} (з мовленням {len(evaluation.positive_runs)}, "
        f"порожніх {len(evaluation.negative_runs)})",
        "",
        "Налаштування (прочитано з UAFastConformer.py):",
        "  " + "  ".join(f"{k}={v}" for k, v in cfg.items() if k not in ("ACTIVE_MODEL", "QUANTIZATION")),
    ]
    if baseline is None:
        lines.append("Baseline ще немає — запустіть з --update-baseline, щоб зафіксувати стартову точку.")
    elif mismatch:
        lines.append(f"⚠ Baseline не порівнюється: {mismatch}. Оновіть: --update-baseline")
    changes = config_changes(evaluation, baseline)
    if changes:
        lines.append("Змінено відносно baseline:")
        lines += [f"  {c}" for c in changes]

    lines += ["", f"{'Кліп':<6} {'Теги':<34} {'CER':>6} {'WER':>6}  {'Оцінка':<10} {'Δ CER':>6}"]
    for run in evaluation.runs:
        clip = run.clip
        tags = ",".join(clip.tags)[:34]
        if clip.expect_empty:
            verdict = "ок (порожньо)" if not normalize(run.recognized) else "✗ зайвий текст"
            lines.append(f"{clip.id:<6} {tags:<34} {'':>6} {'':>6}  {verdict}")
            if normalize(run.recognized):
                lines.append(f"     результат: {run.recognized}")
            continue
        s = evaluation.scores[clip.id]
        delta = _delta(s.cer, base_clips.get(clip.id, {}).get("cer"))
        note = "  (занадто коротко)" if run.too_short else ""
        lines.append(f"{clip.id:<6} {tags:<34} {_pct(s.cer)} {_pct(s.wer)}  {s.grade:<10} {delta:>6}{note}")
        if s.char_errors:
            lines.append(f"     еталон:    {clip.reference}")
            lines.append(f"     результат: {run.recognized}")
            lines.append(f"     різниця:   {s.diff}")

    base_total = comparable.get("total", {}) if comparable else {}
    lines += [
        "",
        f"{'РАЗОМ':<41} {_pct(total.cer)} {_pct(total.wer)}  {'':<10} {_delta(total.cer, base_total.get('cer')):>6}",
    ]
    base_tags = comparable.get("by_tag", {}) if comparable else {}
    for tag, t in evaluation.by_tag().items():
        lines.append(f"  {tag:<39} {_pct(t.cer)} {_pct(t.wer)}  {'':<10} "
                     f"{_delta(t.cer, base_tags.get(tag, {}).get('cer')):>6}")

    gap = evaluation.streaming_gap_cer
    punct = total.punct.f1
    lines += [
        "",
        f"Один прохід на всьому файлі:  CER {_pct(single.cer)}  WER {_pct(single.wer)}",
        f"Стрімінг vs один прохід:      {'—' if gap is None else f'{gap * 100:+.1f}% CER'}"
        "   (частка помилок від розрізів і склейки)",
        f"Пунктуація на межах речень:   {'—' if punct is None else f'{punct * 100:.0f}%'}"
        f"   (влучно {total.punct.tp}, зайвих {total.punct.fp}, пропущених {total.punct.fn})",
    ]
    return "\n".join(lines)


def save_report(dataset_dir: Path, text: str) -> Path:
    folder = Path(dataset_dir) / REPORTS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{dt.datetime.now():%Y-%m-%d_%H%M%S}.txt"
    path.write_text(text + "\n", encoding="utf-8")
    return path
