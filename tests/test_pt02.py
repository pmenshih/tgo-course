"""Тесты работы 2 заочного отделения: читают pt02/artifacts/ и пересчитывают из данных все, что можно пересчитать.

Скрипт студента тесты не запускают и не импортируют, артефакты к их запуску уже должны лежать на месте.
Обязательные артефакты: tiny_config.json, train_curve.csv, samples.json, generation.json, eval.json (bigram и tiny).
Точка qwen в eval.json и артефакты бонусов (greedy_check.json, speed.csv) проверяются, только если они есть;
проверка greedy_check.json загружает Qwen3-0.6B-Base и повторяет жадную генерацию через model.generate.
"""
import csv
import json
import math
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "pt02"
ART = LAB / "artifacts"
DATA = Path(__file__).resolve().parent / "data" / "pt02"
MODEL = os.environ.get("TGO_TEST_MODEL", "Qwen/Qwen3-0.6B-Base")

pytestmark = pytest.mark.skipif(
    "NotImplementedError" in (LAB / "run.py").read_text(encoding="utf-8"),
    reason="pt02/run.py это заглушка, работа не начата"
)


def load_json(path: Path):
    assert path.exists(), f"нет файла {path.relative_to(ROOT)}, запустите python pt02/run.py --fast"
    return json.loads(path.read_text(encoding="utf-8"))


def load_optional(path: Path):
    """Артефакт бонуса: если файла нет, тесты по нему пропускаются."""
    if not path.exists():
        pytest.skip(f"бонус не сделан: нет файла {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path):
    assert path.exists(), f"нет файла {path.relative_to(ROOT)}, запустите python pt02/run.py --fast"
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def thresholds():
    return load_json(DATA / "thresholds.json")


# ---------------------------------------------------------------- tiny_config.json, train_curve.csv

@pytest.fixture(scope="module")
def config():
    return load_json(ART / "tiny_config.json")


@pytest.fixture(scope="module")
def curve():
    rows = load_csv(ART / "train_curve.csv")
    return [{k: float(v) for k, v in r.items()} for r in rows]


def test_config_fields(config):
    for key in ("vocab_size", "d_model", "n_layers", "n_heads", "context", "n_params", "steps", "batch", "lr", "bpe_merges"):
        assert key in config, f"в tiny_config.json нет поля {key}"
    assert config["steps"] >= 200
    assert config["n_layers"] >= 2 and config["n_heads"] >= 2
    assert config["bpe_merges"] >= 200, "слияний из работы 1 меньше 200"


def test_curve_shape(config, curve):
    assert curve[0]["step"] == 0, "первая строка кривой должна быть на шаге 0, до первого обновления"
    steps = [r["step"] for r in curve]
    assert all(a < b for a, b in zip(steps, steps[1:])), "step должен строго расти"
    assert steps[-1] == config["steps"]
    assert len(curve) >= 5


def test_curve_starts_random(config, curve):
    # случайные веса: лосс около логарифма размера словаря
    assert abs(curve[0]["train_loss"] - math.log(config["vocab_size"])) < 1.0


def test_curve_falls(curve, thresholds):
    drop = curve[0]["val_loss"] - curve[-1]["val_loss"]
    assert drop >= thresholds["val_loss_drop"], f"val_loss упал только на {drop:.2f}"
    assert curve[-1]["val_loss"] < curve[-1]["train_loss"] + 1.0, "val_loss сильно выше train_loss: переобучение или ошибка в оценке"


# ---------------------------------------------------------------- samples.json

@pytest.fixture(scope="module")
def samples():
    return load_json(ART / "samples.json")


def test_samples(samples, config):
    rows = samples["samples"]
    assert len(rows) >= 4
    steps = [r["step"] for r in rows]
    assert steps[0] == 0 and steps[-1] == config["steps"]
    assert rows[-1]["text"].strip(), "последний образец пустой"
    assert rows[-1]["text"] != rows[0]["text"]
    assert all(r["text"].startswith(samples["prompt"].split()[0]) for r in rows), "образец должен начинаться с затравки"


# ---------------------------------------------------------------- eval.json

@pytest.fixture(scope="module")
def evaluation():
    return load_json(ART / "eval.json")


@pytest.mark.parametrize("name", ["bigram", "tiny", "qwen"])
def test_eval_consistent(evaluation, name):
    if name == "qwen" and name not in evaluation:
        pytest.skip("бонус не сделан: в eval.json нет точки qwen")
    assert name in evaluation, f"в eval.json нет модели {name}"
    e = evaluation[name]
    assert e["tokens"] > 0
    assert e["nats_per_word"] == pytest.approx(e["nats_total"] / evaluation["words"], rel=1e-3)
    assert e["ppl_per_word"] == pytest.approx(math.exp(e["nats_per_word"]), rel=1e-2)


def test_eval_order(evaluation, config):
    assert evaluation["words"] >= 300
    assert evaluation["bigram"]["alpha"] == pytest.approx(0.01)
    tokens_per_word = evaluation["tiny"]["tokens"] / evaluation["words"]
    uniform = math.log(config["vocab_size"]) * tokens_per_word
    assert evaluation["tiny"]["nats_per_word"] < uniform, "маленькая модель не лучше равновероятного выбора"
    if "qwen" in evaluation:
        assert evaluation["qwen"]["nats_per_word"] < evaluation["tiny"]["nats_per_word"]
        assert evaluation["qwen"]["model"] == MODEL


# ---------------------------------------------------------------- greedy_check.json

@pytest.fixture(scope="module")
def greedy():
    return load_optional(ART / "greedy_check.json")


def test_greedy_recorded(greedy):
    assert greedy["model"] == MODEL
    assert len(greedy["ours_ids"]) >= 20
    assert greedy["ours_ids"] == greedy["hf_ids"], "записанные ours_ids и hf_ids не совпадают"
    assert greedy["match"] is True


def test_greedy_matches_hf(greedy):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    ids = torch.tensor([greedy["prompt_ids"]])
    n = len(greedy["hf_ids"])
    with torch.no_grad():
        out = model.generate(ids, do_sample=False, max_new_tokens=n, pad_token_id=tok.eos_token_id)
    expected = out[0, ids.size(1):].tolist()
    assert greedy["ours_ids"] == expected, "ваша жадная генерация не совпадает с model.generate"


# ---------------------------------------------------------------- generation.json

@pytest.fixture(scope="module")
def generation():
    return load_json(ART / "generation.json")


EXPECTED_MODES = {
    "greedy": (0.0, None),
    "t0.7": (0.7, None),
    "t1.5": (1.5, None),
    "top_p0.9": (1.0, 0.9),
}


def test_generation_modes(generation):
    assert generation["model"] == "tiny", "generation.json должен быть получен на вашей модели, не на Qwen3"
    runs = {r["name"]: r for r in generation["runs"]}
    for name, (t, p) in EXPECTED_MODES.items():
        assert name in runs, f"нет режима {name}"
        r = runs[name]
        assert r["temperature"] == pytest.approx(t)
        if p is None:
            assert r["top_p"] is None
        else:
            assert r["top_p"] == pytest.approx(p)
        assert r["n_tokens"] == len(r["ids"]) >= 1
        assert r["text"].strip()


def first_repeat(ids, n=4):
    seen = set()
    for i in range(n, len(ids) + 1):
        g = tuple(ids[i - n:i])
        if g in seen:
            return i - n
        seen.add(g)
    return None


def test_greedy_loop_start(generation):
    greedy = next(r for r in generation["runs"] if r["name"] == "greedy")
    assert greedy["loop_start"] == first_repeat(greedy["ids"]), "loop_start не совпадает с первой повторившейся четверкой токенов"


# ---------------------------------------------------------------- speed.csv

@pytest.fixture(scope="module")
def speed():
    if not (ART / "speed.csv").exists():
        pytest.skip("бонус не сделан: нет файла pt02/artifacts/speed.csv")
    rows = load_csv(ART / "speed.csv")
    return {r["mode"]: {k: float(v) for k, v in r.items() if k != "mode"} for r in rows}


def test_speed(speed):
    for mode in ("cache", "no_cache"):
        assert mode in speed, f"в speed.csv нет строки {mode}"
        r = speed[mode]
        assert r["tokens_per_second"] == pytest.approx(r["new_tokens"] / r["seconds"], rel=0.02)
    assert speed["cache"]["tokens_per_second"] > speed["no_cache"]["tokens_per_second"], "с кэшем должно быть быстрее"
