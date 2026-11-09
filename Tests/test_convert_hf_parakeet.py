"""Тести App/ConvertHfParakeet.py на маленькій моделі з випадковими вагами.

Архітектура та сама, що в parakeet-tdt-0.6b-v3-uk (1024/640, 8193 токени, 128 mel),
але 2 шари енкодера — тому без інтернету і за ~десяток секунд. Перевіряється
механізм експорту (ONNX == PyTorch, формат для onnx_asr), а не якість розпізнавання.

    uv run pytest Tests/test_convert_hf_parakeet.py -v
    uv run pytest -m export -v          # те саме за маркером
    uv run pytest -m "not export"       # усе, крім цих тестів
"""
import numpy as np
import pytest

pytestmark = pytest.mark.export

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnx_asr")

import json  # noqa: E402

import onnxruntime as rt  # noqa: E402
import torch  # noqa: E402

import ConvertHfParakeet as conv  # noqa: E402

SAMPLE_RATE = conv.SAMPLE_RATE


@pytest.fixture(scope="module")
def random_model():
    return conv.build_random_model(layers=2)


@pytest.fixture(scope="module")
def exported(random_model, tmp_path_factory):
    """Один експорт на весь модуль: (model, feature_extractor, tokenizer, out_dir)."""
    model, fe, tokenizer = random_model
    out_dir = tmp_path_factory.mktemp("parakeet_onnx")
    conv.stage_export(model, out_dir, model.config.encoder_config.num_mel_bins)
    conv.stage_vocab_config(model, tokenizer, out_dir)
    return model, fe, tokenizer, out_dir


def _noise(seconds: float, seed: int) -> np.ndarray:
    return (0.1 * np.random.default_rng(seed).standard_normal(int(SAMPLE_RATE * seconds))).astype(np.float32)


# ---- етап 2: конфігурація -----------------------------------------------------

def test_config_checks_pass(random_model):
    model, fe, _ = random_model
    conv.check_config(model, fe)


def test_config_rejects_wrong_start_token(random_model):
    model, fe, _ = random_model
    old = model.generation_config.decoder_start_token_id
    model.generation_config.decoder_start_token_id = 0
    try:
        with pytest.raises(conv.StageError, match="decoder_start_token_id"):
            conv.check_config(model, fe)
    finally:
        model.generation_config.decoder_start_token_id = old


# ---- етап 3: ONNX-файли ----------------------------------------------------------

def test_onnx_files_created(exported):
    *_, out_dir = exported
    for name in conv.ONNX_FILES:
        assert (out_dir / name).is_file(), name
    assert (out_dir / "encoder-model.onnx.data").is_file(), "ваги енкодера — окремим файлом"


@pytest.mark.parametrize("frames", [120, 517, 1500])
def test_encoder_matches_pytorch_without_padding(exported, frames):
    """Довжина входу відрізняється від тієї, що при експорті (300) — динамічна вісь часу."""
    model, *_, out_dir = exported
    enc_w, _ = conv.make_wrappers(model)
    sess = rt.InferenceSession(str(out_dir / "encoder-model.onnx"), providers=["CPUExecutionProvider"])

    x = np.random.default_rng(frames).standard_normal((1, 128, frames)).astype(np.float32)
    xl = np.array([frames], dtype=np.int64)
    out, out_len = sess.run(["outputs", "encoded_lengths"], {"audio_signal": x, "length": xl})
    with torch.inference_mode():
        ref, ref_len = enc_w(torch.from_numpy(x), torch.from_numpy(xl))

    assert np.isfinite(out).all(), "NaN/inf у виході енкодера"
    assert out_len.tolist() == ref_len.tolist()
    assert out.shape == (1, 1024, int(ref_len[0]))
    np.testing.assert_allclose(out, ref.numpy(), atol=conv.ENCODER_ATOL)


@pytest.mark.parametrize("frames,valid", [(193, 192), (517, 477), (300, 120)])
def test_encoder_padded_input_has_no_nan(exported, frames, valid):
    """Вхід з паддінгом у кінці (кадрів більше, ніж length).

    Саме так onnx_asr подає ознаки: препроцесор nemo128 дає на 1 кадр більше,
    ніж валідна довжина, і для ~12% довжин після стиснення ×8 виникає повністю
    замаскований кадр енкодера. Softmax по рядку з одних −∞ = NaN, який через увагу
    розходиться на всі кадри → декодер видає <unk><unk>… (NaN → argmax = 0)."""
    model, *_, out_dir = exported
    enc_w, _ = conv.make_wrappers(model)
    sess = rt.InferenceSession(str(out_dir / "encoder-model.onnx"), providers=["CPUExecutionProvider"])

    x = np.random.default_rng(frames).standard_normal((1, 128, frames)).astype(np.float32)
    xl = np.array([valid], dtype=np.int64)
    out, out_len = sess.run(["outputs", "encoded_lengths"], {"audio_signal": x, "length": xl})
    with torch.inference_mode():
        ref, ref_len = enc_w(torch.from_numpy(x), torch.from_numpy(xl))

    n = int(ref_len[0])
    assert out_len.tolist() == ref_len.tolist()
    assert np.isfinite(out).all(), f"NaN/inf у виході енкодера: {np.isnan(out).mean():.0%} значень"
    np.testing.assert_allclose(out[..., :n], ref.numpy()[..., :n], atol=conv.ENCODER_ATOL)


def test_onnx_asr_audio_with_padded_encoder_frame(exported):
    """Відтворення реального збою: кліп 1.924 с з VTT_DATASET давав суцільні <unk>.

    Перевіряє весь шлях застосунку (аудіо → препроцесор onnx_asr → енкодер → декодер)."""
    import onnx_asr

    model, fe, tokenizer, out_dir = exported
    asr_model = onnx_asr.load_model("nemo-conformer-tdt", str(out_dir))
    asr = asr_model.asr
    audio = _noise(1.924, seed=3)

    # Передумова: для цієї довжини onnx_asr справді дає зайвий (замаскований) кадр енкодера.
    feats, flen = asr._preprocessor(audio[None], np.array([len(audio)], dtype=np.int64))
    sub = model.encoder._get_subsampling_output_length
    total, valid = int(sub(torch.tensor([feats.shape[2]]))), int(sub(torch.tensor([int(flen[0])])))
    assert (feats.shape[2], int(flen[0]), total, valid) == (193, 192, 25, 24)

    enc_out, enc_len = asr._encode(feats, flen)
    assert np.isfinite(enc_out).all(), "NaN у виході енкодера onnx_asr"

    hf_feats, hf_mask = conv.hf_features(fe, audio)
    ref = " ".join(tokenizer.decode(conv.hf_greedy_ids(model, hf_feats, hf_mask),
                                    skip_special_tokens=True, group_tokens=False).split())
    res = asr_model.with_timestamps().recognize(audio, sample_rate=SAMPLE_RATE)
    assert "<unk>" not in res.text
    assert " ".join(res.text.split()) == ref
    assert all(np.isfinite(res.logprobs))


def test_decoder_joint_matches_pytorch(exported):
    model, *_, out_dir = exported
    _, dec_w = conv.make_wrappers(model)
    sess = rt.InferenceSession(str(out_dir / "decoder_joint-model.onnx"), providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(5)
    feeds = {
        "encoder_outputs": rng.standard_normal((1, 1024, 1)).astype(np.float32),
        "targets": np.array([[model.config.blank_token_id]], dtype=np.int32),   # старт декодування
        "target_length": np.array([1], dtype=np.int32),
        "input_states_1": rng.standard_normal((2, 1, 640)).astype(np.float32),
        "input_states_2": rng.standard_normal((2, 1, 640)).astype(np.float32),
    }
    out = sess.run(["outputs", "prednet_lengths", "output_states_1", "output_states_2"], feeds)
    with torch.inference_mode():
        ref = dec_w(*(torch.from_numpy(v) for v in feeds.values()))

    assert out[0].shape == (1, 1, 1, 8193 + 5)   # vocab + 5 тривалостей TDT
    for got, exp in zip(out, ref, strict=True):
        assert np.isfinite(got).all()
        np.testing.assert_allclose(got, exp.numpy(), atol=conv.DECODER_ATOL)


# ---- етап 4: vocab.txt / config.json ----------------------------------------------

def test_vocab_txt_format(exported):
    model, _, tokenizer, out_dir = exported
    lines = (out_dir / "vocab.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == model.config.vocab_size
    assert lines[-1] == f"<blk> {model.config.blank_token_id}"
    for i, line in enumerate(lines[:-1]):
        token, idx = line.split(" ")
        assert int(idx) == i
        assert token == tokenizer.convert_ids_to_tokens(i)


def test_config_json(exported):
    *_, out_dir = exported
    cfg = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    assert cfg == {"model_type": "nemo-conformer-tdt", "features_size": 128,
                   "subsampling_factor": 8, "max_tokens_per_step": 10}


# ---- етап 5: onnx_asr ----------------------------------------------------------------

def test_stage_onnx_asr_passes(exported):
    """Ознаки, токени на тих самих ознаках, текст і logprobs — усі перевірки етапу 5."""
    model, fe, tokenizer, out_dir = exported
    audio = _noise(3, seed=7)
    conv.stage_onnx_asr(model, fe, tokenizer, out_dir, test_audio=audio, probe_audio=audio)


@pytest.mark.parametrize("seconds", [1.3, 4.0, 9.7])
def test_onnx_asr_text_equals_hf(exported, seconds):
    """Повний шлях застосунку (аудіо → onnx_asr → текст) == model.generate() у HF."""
    import onnx_asr

    model, fe, tokenizer, out_dir = exported
    audio = _noise(seconds, seed=int(seconds * 10))

    feats, mask = conv.hf_features(fe, audio)
    ref = " ".join(tokenizer.decode(conv.hf_greedy_ids(model, feats, mask),
                                    skip_special_tokens=True, group_tokens=False).split())
    res = onnx_asr.load_model("nemo-conformer-tdt", str(out_dir)).with_timestamps().recognize(
        audio, sample_rate=SAMPLE_RATE)

    assert ref, "випадкова модель має видати хоч якийсь текст"
    assert " ".join(res.text.split()) == ref
    assert res.logprobs is not None and len(res.logprobs) == len(res.tokens)
    assert all(np.isfinite(res.logprobs))


def test_stage_onnx_asr_fails_on_empty_dir(random_model, tmp_path):
    model, fe, tokenizer = random_model
    audio = _noise(1, seed=1)
    with pytest.raises(conv.StageError, match="файли для onnx_asr"):
        conv.stage_onnx_asr(model, fe, tokenizer, tmp_path, test_audio=None, probe_audio=audio)


# ---- CLI --------------------------------------------------------------------------------

def test_selftest_cli_returns_zero():
    assert conv.main(["--selftest", "--selftest-layers", "1"]) == 0


def test_missing_test_wav_stops(tmp_path, capsys):
    assert conv.main(["--test-wav", str(tmp_path / "nope.wav"), "--skip-download",
                      "--hf-dir", str(tmp_path)]) == 1
    assert "немає файлу" in capsys.readouterr().out
