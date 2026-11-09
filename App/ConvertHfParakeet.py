"""
ConvertHfParakeet.py — експорт Parakeet TDT з формату Hugging Face Transformers
(model.safetensors) в ONNX-формат, який розуміє onnx_asr ("nemo-conformer-tdt").

Працює на Windows: NeMo НЕ потрібен, лише torch + transformers + onnx.

    uv add transformers torch onnx onnxruntime huggingface_hub librosa soundfile
    uv run python App/ConvertHfParakeet.py --test-wav шлях\\до\\фрази.wav

Етапи (кожен перевіряється; якщо перевірка не пройшла — скрипт зупиняється):
  1. Завантаження файлів моделі з HF (без coreml/*) або з локальної папки.
  2. Завантаження в PyTorch + перевірка конфігурації (+ еталонний текст з --test-wav).
  3. Експорт encoder-model.onnx і decoder_joint-model.onnx; виходи ONNX = виходи PyTorch.
  4. vocab.txt + config.json для onnx_asr.
  5. Перевірка через onnx_asr: ознаки (mel), розпізнавання на тих самих ознаках,
     і (з --test-wav) повний текст onnx_asr == еталон з етапу 2.

--selftest — те саме на маленькій моделі з випадковими вагами (без інтернету):
перевіряє сам механізм експорту, а не якість розпізнавання.

Після успіху в --out лежать: encoder-model.onnx(.data), decoder_joint-model.onnx,
vocab.txt, config.json. Завантаження:
    onnx_asr.load_model("nemo-conformer-tdt", r"models\\parakeet-tdt-0.6b-v3-uk")
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HF_REPO = "ihnatov-yuri/parakeet-tdt-0.6b-v3-uk"
DEFAULT_HF_DIR = ROOT / "models" / "hf" / "parakeet-tdt-0.6b-v3-uk"
DEFAULT_OUT_DIR = ROOT / "models" / "parakeet-tdt-0.6b-v3-uk"

HF_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
ONNX_FILES = ("encoder-model.onnx", "decoder_joint-model.onnx", "vocab.txt", "config.json")
SAMPLE_RATE = 16_000
OPSET = 17

# Допуски порівняння ONNX ↔ PyTorch (fp32, різні ядра обчислень).
ENCODER_ATOL = 1e-3
DECODER_ATOL = 1e-3
FEATURES_ATOL = 1e-2


class StageError(RuntimeError):
    pass


def stage(title: str):
    print(f"\n=== {title} ===", flush=True)


def ok(msg: str):
    print(f"  ✅ {msg}", flush=True)


def check(cond: bool, msg: str):
    if not cond:
        raise StageError(msg)
    ok(msg)


# --------------------------------------------------------------------------
# ONNX-обгортки: імена входів/виходів точно як у NeMo-експорті,
# саме їх читає onnx_asr/models/nemo.py (NemoConformerRnnt / NemoConformerTdt).
# --------------------------------------------------------------------------

def make_wrappers(model):
    import torch
    from torch import nn

    class EncoderWrapper(nn.Module):
        """audio_signal [B, mels, T], length [B]  ->  outputs [B, 1024, T'], encoded_lengths [B]."""

        def __init__(self, m):
            super().__init__()
            self.encoder = m.encoder

        def forward(self, audio_signal, length):
            feats = audio_signal.transpose(1, 2)                       # HF чекає [B, T, mels]
            frames = torch.arange(feats.shape[1], device=feats.device)
            mask = (frames[None, :] < length[:, None]).long()
            out = self.encoder(input_features=feats, attention_mask=mask)
            enc_len = out.attention_mask.sum(-1).to(torch.int64)
            return out.last_hidden_state.transpose(1, 2), enc_len

    class DecoderJointWrapper(nn.Module):
        """Один крок transducer-декодера + joint (як decoder_joint у NeMo).

        encoder_outputs [B, 1024, 1], targets [B, 1], target_length [B],
        input_states_1/2 [layers, B, 640]  ->  outputs [B, 1, 1, vocab+durations],
        prednet_lengths [B], output_states_1/2."""

        def __init__(self, m):
            super().__init__()
            self.encoder_projector = m.encoder_projector
            self.embedding = m.decoder.embedding
            self.lstm = m.decoder.lstm
            self.decoder_projector = m.decoder.decoder_projector
            self.joint = m.joint

        def forward(self, encoder_outputs, targets, target_length, input_states_1, input_states_2):
            emb = self.embedding(targets.long())
            lstm_out, (h, c) = self.lstm(emb, (input_states_1, input_states_2))
            dec = self.decoder_projector(lstm_out)                        # [B, U, 640]
            enc = self.encoder_projector(encoder_outputs.transpose(1, 2))  # [B, T, 640]
            logits = self.joint(
                encoder_hidden_states=enc[:, :, None, :],
                decoder_hidden_states=dec[:, None, :, :],
            )                                                              # [B, T, U, V+D]
            return logits, target_length, h, c

    return EncoderWrapper(model).eval(), DecoderJointWrapper(model).eval()


# --------------------------------------------------------------------------
# Етапи
# --------------------------------------------------------------------------

def stage_download(src: str, hf_dir: Path, skip_download: bool) -> Path:
    stage("1. Файли моделі Hugging Face")
    local = Path(src)
    if local.is_dir():
        hf_dir = local
        print(f"  Локальна папка: {hf_dir}")
    elif not skip_download:
        from huggingface_hub import snapshot_download

        print(f"  Завантажую {src} → {hf_dir} (без coreml/*, ~2.4 ГБ) ...")
        snapshot_download(repo_id=src, local_dir=str(hf_dir), allow_patterns=list(HF_FILES))
    missing = [f for f in HF_FILES if not (hf_dir / f).is_file()]
    check(not missing, f"усі {len(HF_FILES)} файлів на місці ({hf_dir})" if not missing
          else f"бракує файлів у {hf_dir}: {missing}")
    size_gb = (hf_dir / "model.safetensors").stat().st_size / 1024 ** 3
    check(size_gb > 0.5, f"model.safetensors = {size_gb:.2f} ГБ")
    return hf_dir


def stage_load_hf(hf_dir: Path):
    stage("2. Завантаження в PyTorch")
    import torch
    from transformers import AutoProcessor, ParakeetForTDT

    t = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(hf_dir))
    model = ParakeetForTDT.from_pretrained(str(hf_dir), dtype=torch.float32).eval()
    ok(f"модель завантажено за {time.perf_counter() - t:.1f} с")
    return model, processor.feature_extractor, processor.tokenizer, processor


def check_config(model, feature_extractor):
    cfg = model.config
    n_dur = len(cfg.durations)
    # onnx_asr бере step = argmax по хвосту логітів, тобто індекс = тривалість.
    check(list(cfg.durations) == list(range(n_dur)),
          f"durations = {list(cfg.durations)} (onnx_asr вимагає 0..N-1)")
    check(cfg.blank_token_id == cfg.vocab_size - 1,
          f"blank_token_id = {cfg.blank_token_id} — останній токен словника ({cfg.vocab_size})")
    mels = cfg.encoder_config.num_mel_bins
    check(mels in (80, 128), f"num_mel_bins = {mels}")
    check(feature_extractor.feature_size == mels,
          f"feature_extractor.feature_size = {feature_extractor.feature_size} = num_mel_bins")
    check(feature_extractor.sampling_rate == SAMPLE_RATE, f"sampling_rate = {SAMPLE_RATE}")
    sub = cfg.encoder_config.subsampling_factor
    check(sub == 8, f"subsampling_factor = {sub}")
    # onnx_asr починає декодування з blank — HF має стартувати з того самого токена.
    start = getattr(model.generation_config, "decoder_start_token_id", None)
    check(start == cfg.blank_token_id, f"decoder_start_token_id = {start} = blank")


def hf_features(feature_extractor, audio: np.ndarray):
    inputs = feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    return inputs["input_features"], inputs["attention_mask"]


def hf_greedy_ids(model, input_features, attention_mask) -> list[int]:
    import torch

    with torch.inference_mode():
        out = model.generate(input_features=input_features, attention_mask=attention_mask)
    seq = out.sequences if hasattr(out, "sequences") else out
    blank, pad = model.config.blank_token_id, model.config.pad_token_id
    return [int(i) for i in seq[0].tolist() if int(i) not in (blank, pad)]


def stage_export(model, out_dir: Path, mels: int):
    stage("3. Експорт в ONNX")
    import onnx
    import onnxruntime as rt
    import torch

    enc_w, dec_w = make_wrappers(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="onnx_export_"))
    try:
        # --- енкодер --------------------------------------------------------
        T = 300  # 3 с ознак; довжина динамічна
        feats = torch.randn(1, mels, T)
        length = torch.tensor([T], dtype=torch.int64)
        t = time.perf_counter()
        torch.onnx.export(
            enc_w, (feats, length), str(tmp / "encoder-model.onnx"),
            input_names=["audio_signal", "length"],
            output_names=["outputs", "encoded_lengths"],
            dynamic_axes={"audio_signal": {0: "batch", 2: "time"}, "length": {0: "batch"},
                          "outputs": {0: "batch", 2: "enc_time"}, "encoded_lengths": {0: "batch"}},
            opset_version=OPSET, dynamo=False,
        )
        # Ваги енкодера > 2 ГБ: в один .onnx не влазять (ліміт protobuf), тому —
        # граф у encoder-model.onnx, усі ваги одним файлом encoder-model.onnx.data
        # (так само, як в istupakov/parakeet-tdt-0.6b-v3-onnx).
        enc_model = onnx.load(str(tmp / "encoder-model.onnx"), load_external_data=True)
        for p in out_dir.glob("encoder-model.onnx*"):
            p.unlink()
        onnx.save_model(enc_model, str(out_dir / "encoder-model.onnx"), save_as_external_data=True,
                        all_tensors_to_one_file=True, location="encoder-model.onnx.data", size_threshold=1024)
        del enc_model
        ok(f"encoder-model.onnx експортовано за {time.perf_counter() - t:.1f} с")

        # --- decoder + joint ------------------------------------------------
        cfg = model.config
        layers, hid = cfg.num_decoder_layers, cfg.decoder_hidden_size
        enc_hid = cfg.encoder_config.hidden_size
        dec_inputs = (
            torch.randn(1, enc_hid, 1),
            torch.tensor([[5]], dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
            torch.zeros(layers, 1, hid),
            torch.zeros(layers, 1, hid),
        )
        torch.onnx.export(
            dec_w, dec_inputs, str(out_dir / "decoder_joint-model.onnx"),
            input_names=["encoder_outputs", "targets", "target_length", "input_states_1", "input_states_2"],
            output_names=["outputs", "prednet_lengths", "output_states_1", "output_states_2"],
            dynamic_axes={"encoder_outputs": {0: "batch", 2: "enc_time"}, "targets": {0: "batch", 1: "tgt_len"},
                          "target_length": {0: "batch"}, "input_states_1": {1: "batch"},
                          "input_states_2": {1: "batch"}, "outputs": {0: "batch", 1: "enc_time", 2: "tgt_len"},
                          "prednet_lengths": {0: "batch"}, "output_states_1": {1: "batch"},
                          "output_states_2": {1: "batch"}},
            opset_version=OPSET, dynamo=False,
        )
        ok("decoder_joint-model.onnx експортовано")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # --- перевірка: ONNX == PyTorch на іншій довжині, ніж при експорті --------
    for name in ("encoder-model.onnx", "decoder_joint-model.onnx"):
        onnx.checker.check_model(str(out_dir / name))
    ok("onnx.checker: обидві моделі валідні")

    enc_sess = rt.InferenceSession(str(out_dir / "encoder-model.onnx"), providers=["CPUExecutionProvider"])
    dec_sess = rt.InferenceSession(str(out_dir / "decoder_joint-model.onnx"), providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(0)
    # Вхід БЕЗ паддінгу — так само, як його подає onnx_asr (один запис, length = усі кадри).
    # Паддінг у тесті давав повністю замасковані рядки уваги → softmax(−∞…) = NaN
    # у деяких версіях torch-експорту, хоча в застосунку такого входу не буває.
    # Довжина 517 ≠ 300 (як при експорті) — перевіряє, що динамічна вісь часу працює.
    T2 = 517
    x = rng.standard_normal((1, mels, T2)).astype(np.float32)
    xl = np.array([T2], dtype=np.int64)
    with torch.inference_mode():
        ref_out, ref_len = enc_w(torch.from_numpy(x), torch.from_numpy(xl))
    onnx_out, onnx_len = enc_sess.run(["outputs", "encoded_lengths"], {"audio_signal": x, "length": xl})
    check(onnx_len.tolist() == ref_len.tolist(), f"encoded_lengths збігаються: {onnx_len.tolist()}")
    check(bool(np.isfinite(onnx_out).all()), "у виходах енкодера ONNX немає NaN/inf")
    diff = float(np.abs(onnx_out - ref_out.numpy()).max())
    check(diff < ENCODER_ATOL, f"енкодер: max|ONNX − PyTorch| = {diff:.2e} (< {ENCODER_ATOL})")

    frame = onnx_out[0, :, 3]
    s1 = rng.standard_normal((layers, 1, hid)).astype(np.float32)
    s2 = rng.standard_normal((layers, 1, hid)).astype(np.float32)
    feeds = {"encoder_outputs": frame[None, :, None], "targets": np.array([[42]], dtype=np.int32),
             "target_length": np.array([1], dtype=np.int32), "input_states_1": s1, "input_states_2": s2}
    o, _, h, c = dec_sess.run(["outputs", "prednet_lengths", "output_states_1", "output_states_2"], feeds)
    with torch.inference_mode():
        ro, _, rh, rc = dec_w(*(torch.from_numpy(v) for v in feeds.values()))
    expected_width = cfg.vocab_size + len(cfg.durations)
    check(o.shape[-1] == expected_width, f"ширина логітів = {o.shape[-1]} (vocab {cfg.vocab_size} + "
                                         f"{len(cfg.durations)} тривалостей)")
    d = max(float(np.abs(o - ro.numpy()).max()), float(np.abs(h - rh.numpy()).max()),
            float(np.abs(c - rc.numpy()).max()))
    check(d < DECODER_ATOL, f"decoder_joint: max|ONNX − PyTorch| = {d:.2e} (< {DECODER_ATOL})")


def stage_vocab_config(model, tokenizer, out_dir: Path):
    stage("4. vocab.txt і config.json")
    cfg = model.config
    blank = cfg.blank_token_id
    tokens = tokenizer.convert_ids_to_tokens(list(range(blank)))
    bad = [(i, t) for i, t in enumerate(tokens) if t is None or " " in t or "\n" in t]
    check(not bad, "у токенах немає пробілів/переносів (формат vocab.txt «токен id»)" if not bad
          else f"токени з пробілами: {bad[:5]}")
    with (out_dir / "vocab.txt").open("wt", encoding="utf-8", newline="\n") as f:
        for i, tok in enumerate(tokens):
            f.write(f"{tok} {i}\n")
        f.write(f"<blk> {blank}\n")

    config = {
        "model_type": "nemo-conformer-tdt",
        "features_size": cfg.encoder_config.num_mel_bins,
        "subsampling_factor": cfg.encoder_config.subsampling_factor,
        "max_tokens_per_step": cfg.max_symbols_per_step,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")

    lines = (out_dir / "vocab.txt").read_text(encoding="utf-8").splitlines()
    check(len(lines) == cfg.vocab_size, f"vocab.txt: {len(lines)} рядків = vocab_size")
    check(lines[-1] == f"<blk> {blank}", f"останній рядок: «{lines[-1]}»")
    ok(f"config.json: {config}")


def stage_onnx_asr(model, feature_extractor, tokenizer, out_dir: Path, test_audio: np.ndarray | None,
                   probe_audio: np.ndarray):
    stage("5. Перевірка через onnx_asr")
    import onnx_asr

    missing = [f for f in ONNX_FILES if not (out_dir / f).is_file()]
    check(not missing, f"усі файли для onnx_asr на місці: {', '.join(ONNX_FILES)}")

    asr_model = onnx_asr.load_model("nemo-conformer-tdt", str(out_dir), providers=["CPUExecutionProvider"])
    ok("onnx_asr.load_model('nemo-conformer-tdt', ...) — завантажено")
    asr = asr_model.asr

    # 5a. Ознаки: вбудований препроцесор onnx_asr (nemo128) ≈ feature extractor HF.
    audio = test_audio if test_audio is not None else probe_audio
    feats_pt, mask_pt = hf_features(feature_extractor, audio)
    hf_f = feats_pt[0].numpy().T                                      # [mels, T]
    oa_f, oa_len = asr._preprocessor(audio[None, :].astype(np.float32), np.array([len(audio)], dtype=np.int64))
    n = min(int(mask_pt.sum()), int(oa_len[0]))
    check(abs(int(mask_pt.sum()) - int(oa_len[0])) <= 1,
          f"кількість кадрів ознак: HF {int(mask_pt.sum())}, onnx_asr {int(oa_len[0])}")
    fdiff = float(np.abs(oa_f[0, :, :n] - hf_f[:, :n]).mean())
    check(fdiff < FEATURES_ATOL, f"ознаки mel: середня |onnx_asr − HF| = {fdiff:.2e} (< {FEATURES_ATOL})")

    # 5b. Декодування на ТИХ САМИХ ознаках: цикл onnx_asr == model.generate() у HF.
    hf_ids = hf_greedy_ids(model, feats_pt, mask_pt)
    # HF додає в кінець зайвий кадр-паддінг (mask=0). В ONNX подаємо лише валідні кадри —
    # без паддінгу, як onnx_asr у застосунку (паддінг дає NaN, див. перевірку енкодера).
    valid = int(mask_pt.sum())
    enc_out, enc_len = asr._encode(np.ascontiguousarray(hf_f[None, :, :valid], dtype=np.float32),
                                   np.array([valid], dtype=np.int64))
    check(bool(np.isfinite(enc_out).all()), "у виходах енкодера onnx_asr немає NaN/inf")
    oa_ids = [int(i) for i in next(iter(asr._decoding(enc_out, enc_len)))[0]]
    same = oa_ids == hf_ids
    check(same, f"токени onnx_asr == токени HF ({len(hf_ids)} шт.)" if same
          else f"токени різняться: HF {hf_ids[:20]} … vs onnx_asr {oa_ids[:20]} …")

    # 5c. Повний шлях як у застосунку (аудіо → текст) — тільки з реальною фразою.
    if test_audio is not None:
        ref = " ".join(tokenizer.decode(hf_ids, skip_special_tokens=True, group_tokens=False).split())
        res = asr_model.with_timestamps().recognize(test_audio, sample_rate=SAMPLE_RATE)
        got = " ".join(res.text.split())
        print(f"  HF:       {ref!r}\n  onnx_asr: {got!r}")
        check(bool(ref), "еталон HF не порожній")
        check(got == ref, "текст onnx_asr збігається з еталоном HF")
        check(res.logprobs is not None and len(res.logprobs) > 0, "onnx_asr повертає logprobs (фільтр впевненості працює)")


def quantize(out_dir: Path):
    stage("6. Квантизація int8 (опційно)")
    from onnxruntime.quantization import QuantType, quantize_dynamic

    for name in ("encoder-model", "decoder_joint-model"):
        src, dst = out_dir / f"{name}.onnx", out_dir / f"{name}.int8.onnx"
        quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
        ok(f"{dst.name}: {dst.stat().st_size / 1024 ** 2:.0f} МБ")


# --------------------------------------------------------------------------
# Самоперевірка на випадкових вагах (без інтернету)
# --------------------------------------------------------------------------

def build_random_model(layers: int):
    """Модель тієї ж архітектури, що parakeet-tdt-0.6b-v3 (1024/640/8193/128 mel),
    але з випадковими вагами і меншою кількістю шарів енкодера."""
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import ParakeetFeatureExtractor, ParakeetForTDT, ParakeetTDTConfig, ParakeetTokenizer

    torch.manual_seed(0)
    cfg = ParakeetTDTConfig(encoder_config={"num_hidden_layers": layers, "num_mel_bins": 128,
                                            "hidden_size": 1024, "num_attention_heads": 8,
                                            "intermediate_size": 4096})
    model = ParakeetForTDT(cfg).eval()
    # У реальній моделі це задає generation_config.json.
    model.generation_config.decoder_start_token_id = cfg.blank_token_id
    model.generation_config.pad_token_id = cfg.pad_token_id
    # Випадкова ініціалізація дає майже рівні логіти — підсилюємо joint-голову,
    # щоб у тесті модель реально видавала токени, а не лише blank.
    with torch.no_grad():
        model.joint.head.weight.mul_(20.0)
        model.joint.head.bias[cfg.blank_token_id] -= 2.0
        # Як у parakeet-tdt-0.6b-v3-uk: <unk> (id 0) заблоковано bias -1000.
        # Тоді <unk> на виході може з'явитися лише від NaN (argmax по NaN = 0).
        model.joint.head.bias[0] = -1000.0
        # Тривалість 0 для небланкового токена у випадковій моделі тягнеться нескінченно;
        # onnx_asr (як і NeMo) обмежує це max_tokens_per_step, а HF generate() — ні,
        # тож на такому виродженому вході цикли розходяться. Навчена модель так не робить.
        model.joint.head.bias[cfg.vocab_size] -= 50.0

    vocab = [("<unk>", 0.0), ("<s>", 0.0), ("<pad>", 0.0)] + [(f"▁t{i}", -1.0) for i in range(3, cfg.blank_token_id)]
    tk = Tokenizer(models.Unigram(vocab, unk_id=0))
    tk.pre_tokenizer = pre_tokenizers.Metaspace()
    tk.decoder = decoders.Metaspace()
    tokenizer = ParakeetTokenizer(tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>")
    fe = ParakeetFeatureExtractor(feature_size=128)
    return model, fe, tokenizer


def run_selftest(layers: int) -> int:
    out_dir = Path(tempfile.mkdtemp(prefix="parakeet_selftest_"))
    print(f"САМОПЕРЕВІРКА: випадкові ваги, {layers} шар(и) енкодера, вихід → {out_dir}")
    try:
        stage("2. Модель (випадкові ваги)")
        model, fe, tokenizer = build_random_model(layers)
        check_config(model, fe)
        stage_export(model, out_dir, model.config.encoder_config.num_mel_bins)
        stage_vocab_config(model, tokenizer, out_dir)
        rng = np.random.default_rng(1)
        probe = (0.1 * rng.standard_normal(SAMPLE_RATE * 3)).astype(np.float32)
        stage_onnx_asr(model, fe, tokenizer, out_dir, test_audio=None, probe_audio=probe)
    except StageError as e:
        print(f"\n❌ {e}")
        return 1
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
    print("\n✅ Самоперевірка пройдена: механізм експорту працює.")
    return 0


def load_wav(path: Path) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio.astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=HF_REPO, help="repo id на Hugging Face або локальна папка з файлами HF")
    ap.add_argument("--hf-dir", type=Path, default=DEFAULT_HF_DIR, help="куди качати файли HF")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="папка для ONNX-моделі")
    ap.add_argument("--test-wav", type=Path, help="wav з українською фразою для звірки тексту HF ↔ onnx_asr")
    ap.add_argument("--skip-download", action="store_true", help="не качати, файли вже в --hf-dir")
    ap.add_argument("--quantize", action="store_true", help="додатково зробити *.int8.onnx")
    ap.add_argument("--selftest", action="store_true", help="перевірити експорт на випадкових вагах")
    ap.add_argument("--selftest-layers", type=int, default=2)
    args = ap.parse_args(argv)

    if args.selftest:
        return run_selftest(args.selftest_layers)

    try:
        test_audio = None
        if args.test_wav is not None:
            if not args.test_wav.is_file():
                raise StageError(f"немає файлу {args.test_wav}")
            test_audio = load_wav(args.test_wav)

        hf_dir = stage_download(args.src, args.hf_dir, args.skip_download)
        model, fe, tokenizer, _ = stage_load_hf(hf_dir)
        check_config(model, fe)
        if test_audio is not None:
            feats, mask = hf_features(fe, test_audio)
            ref = tokenizer.decode(hf_greedy_ids(model, feats, mask), skip_special_tokens=True, group_tokens=False)
            print(f"  Еталон HF: {ref!r}")
            check(bool(ref.strip()), "еталонний текст не порожній")
        else:
            print("  (без --test-wav: звірки тексту на реальній фразі не буде)")

        stage_export(model, args.out, model.config.encoder_config.num_mel_bins)
        stage_vocab_config(model, tokenizer, args.out)
        rng = np.random.default_rng(1)
        probe = (0.05 * rng.standard_normal(SAMPLE_RATE * 3)).astype(np.float32)
        stage_onnx_asr(model, fe, tokenizer, args.out, test_audio, probe)
        if args.quantize:
            quantize(args.out)
    except StageError as e:
        print(f"\n❌ Зупинено: {e}")
        return 1

    print(f"\n✅ Готово: {args.out}")
    for p in sorted(args.out.iterdir()):
        print(f"   {p.name}  ({p.stat().st_size / 1024 ** 2:.1f} МБ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
