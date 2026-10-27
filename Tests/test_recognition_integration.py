"""
Інтеграційні тести розпізнавання: справжній DictationApp, справжня модель,
справжній run_dictation_session() на ваших записах з VTT_DATASET/clips.

Запуск (з кореня проєкту VoiceToText, поруч з pytest.ini):
    uv run pytest -m integration                      # перевірка
    uv run pytest -m integration --update-baseline    # зафіксувати нову базову лінію

Що підмінено і чому — див. Tests/recognition/harness.py. Метрики — Tests/recognition/asr_metrics.py.
Звіт друкується в кінці прогону і зберігається в VTT_DATASET/reports/.
"""
from __future__ import annotations

import pytest

from App.CaptureDataset_ForTests.dataset_capture import dataset_dir_from_env
from recognition.report import (REPORT_KEY, Evaluation, baseline_mismatch, build_report,
                          load_baseline, save_baseline, save_report)

pytestmark = pytest.mark.integration

# Допуски регресії відносно baseline (абсолютні частки).
CER_TOLERANCE = 0.02
WER_TOLERANCE = 0.03
TAG_CER_TOLERANCE = 0.04
# Наскільки потоковий цикл може бути гіршим за один прохід по всьому файлу.
STREAMING_GAP_LIMIT = 0.03

# --------------------------------------------------------------------------
# Фікстури (усе — один раз на прогін)
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def dataset_dir():

    path = dataset_dir_from_env()
    if path is None:
        pytest.skip("змінна VTT_DATASET не задана")
    if not path.is_dir():
        pytest.skip(f"папки датасету немає: {path}")
    return path


@pytest.fixture(scope="session")
def clips(dataset_dir):
    from recognition.harness import load_dataset
    found = load_dataset(dataset_dir)
    if not found:
        pytest.skip("немає розмічених кліпів (потрібні пари NNNN.wav + NNNN.txt у clips/)")
    return found


@pytest.fixture(scope="session")
def test_app():
    from recognition.harness import TestApp
    try:
        return TestApp()
    except Exception as exc:  # noqa: BLE001 — немає моделі, Qt тощо
        pytest.skip(f"не вдалося підготувати DictationApp / модель: {exc!r}")


@pytest.fixture(scope="session")
def evaluation(request, test_app, clips, dataset_dir) -> Evaluation:
    runs = []
    for clip in clips:
        run = test_app.run_clip(clip)
        if not clip.expect_empty:
            run.single_pass = test_app.single_pass(clip)
        runs.append(run)
    result = Evaluation(test_app.config(), runs)

    baseline = load_baseline(dataset_dir)
    mismatch = baseline_mismatch(result, baseline) if baseline else None
    text = build_report(result, baseline, mismatch)
    report_path = save_report(dataset_dir, text)
    text += f"\n\nЗвіт збережено: {report_path}"
    if request.config.getoption("--update-baseline"):
        text += f"\nBaseline оновлено: {save_baseline(dataset_dir, result)}"
    request.config.stash[REPORT_KEY] = text
    return result


@pytest.fixture(scope="session")
def baseline(request, evaluation, dataset_dir) -> dict:
    """Baseline, з яким коректно порівнювати, — інакше skip з причиною."""
    if request.config.getoption("--update-baseline"):
        pytest.skip("baseline щойно оновлено — порівнювати нема з чим")
    data = load_baseline(dataset_dir)
    if data is None:
        pytest.skip("baseline.json ще немає — запустіть з --update-baseline")
    mismatch = baseline_mismatch(evaluation, data)
    if mismatch:
        pytest.skip(f"{mismatch} — запустіть з --update-baseline")
    return data


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


# --------------------------------------------------------------------------
# Тести
# --------------------------------------------------------------------------

def test_cer_no_regression(evaluation, baseline):
    now, before = evaluation.total.cer, baseline["total"]["cer"]
    if now is None or before is None:
        pytest.skip("немає кліпів з мовленням")
    worst = sorted(((s.cer - baseline["clips"][cid]["cer"], cid) for cid, s in evaluation.scores.items()),
                   reverse=True)[:3]
    assert now <= before + CER_TOLERANCE, (
        f"CER {_pct(now)} > baseline {_pct(before)} + {_pct(CER_TOLERANCE)} допуску; "
        f"найбільше погіршились: " + ", ".join(f"{cid} ({d * 100:+.1f}%)" for d, cid in worst if d > 0))


def test_wer_no_regression(evaluation, baseline):
    now, before = evaluation.total.wer, baseline["total"]["wer"]
    if now is None or before is None:
        pytest.skip("немає кліпів з мовленням")
    assert now <= before + WER_TOLERANCE, (
        f"WER {_pct(now)} > baseline {_pct(before)} + {_pct(WER_TOLERANCE)} допуску")


def test_cer_per_tag_no_regression(evaluation, baseline):
    worse = []
    for tag, totals in evaluation.by_tag().items():
        before = baseline.get("by_tag", {}).get(tag, {}).get("cer")
        if totals.cer is None or before is None:
            continue
        if totals.cer > before + TAG_CER_TOLERANCE:
            worse.append(f"{tag}: {_pct(totals.cer)} (було {_pct(before)})")
    assert not worse, "погіршились теги: " + "; ".join(worse)


def test_negative_clips_empty(evaluation):
    if not evaluation.negative_runs:
        pytest.skip("немає негативних кліпів (порожніх .txt)")
    failed = evaluation.failed_negatives()
    assert not failed, "на тиші/шумі з'явився текст: " + "; ".join(
        f"{r.clip.id}: {r.recognized!r}" for r in failed)


def test_streaming_vs_single_pass(evaluation):
    gap = evaluation.streaming_gap_cer
    if gap is None:
        pytest.skip("немає кліпів з мовленням")
    assert gap <= STREAMING_GAP_LIMIT, (
        f"потоковий цикл гірший за один прохід на {_pct(gap)} CER (ліміт {_pct(STREAMING_GAP_LIMIT)}): "
        f"розрізи або склейка гублять/дублюють слова")


def test_events_order(evaluation):
    problems = []
    for run in evaluation.runs:
        if run.too_short:
            continue
        ev = run.events
        if ev.count("final_audio") != 1 or ev.count("final_done") != 1:
            problems.append(f"{run.clip.id}: final_audio×{ev.count('final_audio')}, final_done×{ev.count('final_done')}")
            continue
        first_final = ev.index("final_audio")
        if "partial_text" in ev[first_final:]:
            problems.append(f"{run.clip.id}: partial_text після final_audio")
        if ev[-1] != "final_done":
            problems.append(f"{run.clip.id}: остання подія {ev[-1]!r}, а не final_done")
    assert not problems, "; ".join(problems)
