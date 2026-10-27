"""
conftest.py — налаштування pytest для папки Tests (додано для інтеграційних тестів розпізнавання).

  маркер integration    — тести на справжній моделі й ваших записах
  --update-baseline     — перезаписати VTT_DATASET/baseline.json поточними результатами

    uv run pytest -m "not integration"                # швидкі тести
    uv run pytest -m integration                      # точність розпізнавання
    uv run pytest -m integration --update-baseline    # нова базова лінія
"""
from __future__ import annotations


def pytest_addoption(parser):
    parser.addoption(
        "--update-baseline", action="store_true", default=False,
        help="перезаписати baseline.json результатами цього прогону (інтеграційні тести)",
    )

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    from recognition.report import REPORT_KEY

    text = config.stash.get(REPORT_KEY, None)
    if text:
        terminalreporter.write_sep("=", "ЗВІТ РОЗПІЗНАВАННЯ")
        for line in text.splitlines():
            terminalreporter.write_line(line)
