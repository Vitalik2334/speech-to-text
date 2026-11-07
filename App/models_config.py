import sys
from enum import Enum
from pathlib import Path


class ModelName(str, Enum):
    UA_FASTCONFORMER = "ua-fastconformer"
    NEMO_CANARY = "nemo-canary-1b-v2"
    NEMO_PARAKEET = "nemo-parakeet-tdt-0.6b-v3"
    # ihnatov-yuri/parakeet-tdt-0.6b-v3-uk, сконвертована в ONNX скриптом App/ConvertHfParakeet.py
    PARAKEET_UK = "parakeet-tdt-0.6b-v3-uk"
    WHISPER_BASE = "whisper-base"
    WHISPER_SMALL = "whisper-small"
    WHISPER_LARGE_V3 = "whisper-large-v3"


MODEL_TYPES: dict[ModelName, str] = {
    ModelName.UA_FASTCONFORMER: "nemo-conformer-rnnt",
    ModelName.NEMO_CANARY: "nemo-canary-1b-v2",
    ModelName.NEMO_PARAKEET: "nemo-parakeet-tdt-0.6b-v3",
    ModelName.PARAKEET_UK: "nemo-conformer-tdt",                         # тип; файли — у MODEL_PATHS
    ModelName.WHISPER_BASE: "whisper-base",                              # короткий alias працює
    ModelName.WHISPER_SMALL: "onnx-community/whisper-small",             # потрібен повний repo id
    ModelName.WHISPER_LARGE_V3: "onnx-community/whisper-large-v3-turbo",  # повний repo id
}

# Рекомендація для CPU-режиму (FORCE_CPU=True) — див. пояснення в докстрінгу
# вище. None означає "лишити fp32, int8 тут не допомагає або шкодить".
RECOMMENDED_CPU_QUANTIZATION: dict[ModelName, str | None] = {
    ModelName.UA_FASTCONFORMER: None,
    ModelName.NEMO_PARAKEET: None,
    ModelName.PARAKEET_UK: None,          # та сама архітектура, що й NEMO_PARAKEET
    ModelName.NEMO_CANARY: "int8",
    ModelName.WHISPER_BASE: "int8",
    ModelName.WHISPER_SMALL: "int8",
    ModelName.WHISPER_LARGE_V3: "int8",
}

# Whisper багатомовний — примусово задаємо мову, щоб не "з'їжджав"
# в іншу мову на короткому/тихому аудіо. Для NeMo-моделей (fastconformer,
# canary, parakeet) параметр language при recognize() передавати не треба.
WHISPER_MODELS = {
    ModelName.WHISPER_BASE,
    ModelName.WHISPER_SMALL,
    ModelName.WHISPER_LARGE_V3,
}
WHISPER_LANGUAGE = "uk"

# Локальні моделі: папка models/ у корені проєкту, а в зібраному exe — поруч з exe.
MODELS_DIR = (
    Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent
) / "models"

# None — onnx_asr сам качає модель з Hugging Face за назвою з MODEL_TYPES.
# Шлях — модель уже лежить локально (MODEL_TYPES тоді містить тип моделі).
MODEL_PATHS: dict[ModelName, Path | None] = {
    ModelName.UA_FASTCONFORMER: None,
    ModelName.NEMO_CANARY: None,
    ModelName.NEMO_PARAKEET: None,
    ModelName.PARAKEET_UK: MODELS_DIR / "parakeet-tdt-0.6b-v3-uk",
    ModelName.WHISPER_BASE: None,
    ModelName.WHISPER_SMALL: None,
    ModelName.WHISPER_LARGE_V3: None,
}