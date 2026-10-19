import sys

from numpy.f2py.auxfuncs import throw_error

from UAFastConformer import NEMO_REPO, MODEL_DIR, MODEL_TYPES, EXPECTED_FILES, ModelName
def convert(quantize: bool) -> int:
    try:
        import nemo.collections.asr as nemo_asr  # noqa: PLC0415
    except ImportError:
        print(
            "NeMo Toolkit не встановлено. Конвертація можлива тільки там, де є\n"
            "nemo_toolkit (Google Colab, WSL2, Linux) — НЕ на голому Windows.\n\n"
            "    pip install \"nemo_toolkit[asr]\"\n\n"
            "Після конвертації скопіюйте папку models/ua-fastconformer на Windows\n"
            "і запустіть цей файл без --convert для самої диктовки."
        )
        return 1

    print(f"Завантажую {NEMO_REPO} через NeMo (перший раз якийсь час) ...")
    model = nemo_asr.models.ASRModel.from_pretrained(NEMO_REPO)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Експортую в ONNX (RNN-T декодер) ...")
    model.export(str(MODEL_DIR / "model.onnx"))
    # NeMo сам розкладає RNN-T на дві частини:
    #   model.onnx -> encoder-model.onnx + decoder_joint-model.onnx
    # це саме ті імена, які очікує onnx-asr — нічого перейменовувати не треба.

    vocab_path = MODEL_DIR / "vocab.txt"
    with vocab_path.open("wt", encoding="utf-8") as f:
        for i, token in enumerate([*model.tokenizer.vocab, "<blk>"]):
            f.write(f"{token} {i}\n")

    config_path = MODEL_DIR / "config.json"
    config_path.write_text(
        '{\n'
        f'    "model_type": "{MODEL_TYPES[ModelName.UA_FASTCONFORMER]}",\n'
        '    "features_size": 80,\n'
        '    "subsampling_factor": 8,\n'
        '    "max_tokens_per_step": 10\n'
        '}\n',
        encoding="utf-8",
    )

    missing = [name for name in EXPECTED_FILES if not (MODEL_DIR / name).exists()]
    if missing:
        print(f"\nУВАГА: NeMo не створив очікувані файли: {missing}")
        print(f"Перевірте вміст {MODEL_DIR} вручну — можливо, змінилась назва у новій версії NeMo.")
        return 1

    print(f"\nГотово: {MODEL_DIR}")
    for name in EXPECTED_FILES:
        size_mb = (MODEL_DIR / name).stat().st_size / 1024 / 1024
        print(f"  {name}  ({size_mb:.1f} МБ)")

    if quantize:
        quantize_model()

    print(f"\nСкопіюйте папку {MODEL_DIR.name}/ на Windows поряд зі скриптом,")
    print("далі запускайте вже без --convert.")
    return 0


def quantize_model() -> None:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: PLC0415
    except ImportError:
        print("\npip install onnxruntime, щоб мати onnxruntime.quantization — пропускаю int8.")
        return

    print("\nКвантизація в int8 (опційно; для UA FastConformer/Parakeet за бенчмарками")
    print("onnx-asr це НЕ прискорює CPU — робіть тільки якщо хочете менший розмір файлу) ...")
    for name in ("encoder-model.onnx", "decoder_joint-model.onnx"):
        src = MODEL_DIR / name
        dst = MODEL_DIR / name.replace(".onnx", ".int8.onnx")
        try:
            quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
            print(f"  {dst.name} ({dst.stat().st_size / 1024 / 1024:.1f} МБ)")
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: квантизація не вдалася ({e}) — лишаю fp32-версію як є.")


def main() -> int:
    args = sys.argv[1:]

    if "--convert" in args:
        return convert(quantize="--quantize" in args)
    else :
        return throw_error()


if __name__ == "__main__":
    raise SystemExit(main())