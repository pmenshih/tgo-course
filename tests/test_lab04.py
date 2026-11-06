"""Тесты лабораторной 4: читают lab04/artifacts/ и пересчитывают из данных все, что можно пересчитать.

Скрипт студента тесты не запускают и не импортируют, артефакты к их запуску уже должны лежать на месте
(адаптер и журнал обучения из train.ipynb, metrics.json и predictions.csv из run.py).
"""
import csv
import json
import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "lab04"
ART = LAB / "artifacts"
DATA = Path(__file__).resolve().parent / "data" / "lab04"
CLASSES = [1, 2, 3, 4, 5]
MODES = ["zero_shot", "few_shot", "adapter"]
TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}

pytestmark = pytest.mark.skipif(
    "NotImplementedError" in (LAB / "run.py").read_text(encoding="utf-8"),
    reason="lab04/run.py это заглушка, работа не начата"
)


def load_json(path: Path):
    assert path.exists(), f"нет файла {path.relative_to(ROOT)}"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def metrics():
    return load_json(ART / "metrics.json")


@pytest.fixture(scope="module")
def predictions():
    path = ART / "predictions.csv"
    assert path.exists(), "нет lab04/artifacts/predictions.csv, запустите python lab04/run.py --fast"
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows, "predictions.csv пуст"
    return rows


@pytest.fixture(scope="module")
def train_config():
    return load_json(ART / "train_config.json")


@pytest.fixture(scope="module")
def thresholds():
    return load_json(DATA / "thresholds.json")


def recompute(rows, mode):
    labels = [int(r["label"]) for r in rows]
    preds = [int(r[mode]) if r[mode].strip() else None for r in rows]
    n = len(labels)
    acc = sum(1 for y, p in zip(labels, preds) if p == y) / n
    f1 = []
    for c in CLASSES:
        tp = sum(1 for y, p in zip(labels, preds) if y == c and p == c)
        fp = sum(1 for y, p in zip(labels, preds) if y != c and p == c)
        fn = sum(1 for y, p in zip(labels, preds) if y == c and p != c)
        f1.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    valid = [(y, p) for y, p in zip(labels, preds) if p is not None]
    mae = sum(abs(y - p) for y, p in valid) / len(valid) if valid else None
    return {"accuracy": acc, "f1_macro": sum(f1) / len(CLASSES), "mae_stars": mae, "invalid": n - len(valid)}


# ---------------------------------------------------------------- predictions.csv и metrics.json

def test_predictions_schema(predictions):
    assert len(predictions) >= 24, "в predictions.csv меньше 24 строк"
    for m in MODES:
        assert m in predictions[0], f"в predictions.csv нет колонки {m}: все три режима обязательны"
    for r in predictions:
        assert int(r["label"]) in CLASSES
        for m in MODES:
            assert r[m].strip() == "" or int(r[m]) in CLASSES, f"предсказание {m} вне 1–5: {r[m]!r}"


def test_class_counts(metrics, predictions):
    counts = {str(c): sum(1 for r in predictions if int(r["label"]) == c) for c in CLASSES}
    assert metrics["class_counts"] == counts, "class_counts в metrics.json не совпадает с колонкой label"
    assert metrics["n_test"] == len(predictions)


@pytest.mark.parametrize("mode", MODES)
def test_metrics_recomputed(metrics, predictions, mode):
    assert mode in metrics["modes"], f"в metrics.json нет режима {mode}"
    got = metrics["modes"][mode]
    ref = recompute(predictions, mode)
    for key in ("accuracy", "f1_macro"):
        assert got[key] == pytest.approx(ref[key], abs=2e-3), f"{mode}: {key} в metrics.json {got[key]}, по predictions.csv {ref[key]:.4f}"
    assert got["invalid"] == ref["invalid"]
    if ref["mae_stars"] is not None:
        assert got["mae_stars"] == pytest.approx(ref["mae_stars"], abs=2e-3)
    assert got["sec_per_example"] > 0


def test_modes_differ(predictions):
    cols = {m: [r[m] for r in predictions] for m in MODES}
    assert not (cols["zero_shot"] == cols["few_shot"] == cols["adapter"]), "три режима дали одинаковые ответы на всех примерах"
    assert len(set(cols["adapter"])) >= 2, "адаптер отвечает одной и той же цифрой на всех примерах"


def test_adapter_beats_zero_shot(metrics, thresholds):
    gain = metrics["modes"]["adapter"]["f1_macro"] - metrics["modes"]["zero_shot"]["f1_macro"]
    key = "f1_macro_gain_fast" if metrics.get("fast") else "f1_macro_gain_full"
    assert gain >= thresholds[key], f"адаптер лучше zero-shot по макро-F1 на {gain:.3f}, порог {thresholds[key]}"
    assert metrics["modes"]["adapter"]["invalid"] <= metrics["modes"]["zero_shot"]["invalid"], "после дообучения ответов без цифры стало больше"


# ---------------------------------------------------------------- адаптер и обучение

def safetensors_numel(path: Path) -> int:
    """Число чисел во всех тензорах по заголовку файла, без загрузки весов."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    total = 0
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        k = 1
        for d in meta["shape"]:
            k *= d
        total += k
    return total


def test_adapter_files(train_config):
    cfg = load_json(ART / "adapter" / "adapter_config.json")
    assert cfg["r"] == 8, f"ранг адаптера {cfg['r']}, в задании 8"
    assert set(cfg["target_modules"]) == TARGETS, f"целевые модули {sorted(cfg['target_modules'])}, нужны все семь"
    weights = ART / "adapter" / "adapter_model.safetensors"
    assert weights.exists(), "нет adapter_model.safetensors: адаптер не сохранен через save_pretrained"
    numel = safetensors_numel(weights)
    assert numel == train_config["trainable_params"], f"в файле адаптера {numel:,} параметров, в train_config.json {train_config['trainable_params']:,}"
    assert train_config["trainable_pct"] == pytest.approx(100 * numel / train_config["total_params"], abs=0.01)
    assert train_config["trainable_pct"] < 2.0, "обучаемых параметров больше двух процентов: это не адаптер ранга 8"


def test_train_config(train_config):
    assert train_config["train_examples"] >= 40
    assert train_config["steps"] >= 5
    assert train_config["lora_r"] == 8 and train_config["lora_alpha"] == 16
    assert set(train_config["target_modules"]) == TARGETS
    assert train_config["train_minutes"] > 0


def test_train_log_falls():
    path = ART / "train_log.csv"
    assert path.exists(), "нет lab04/artifacts/train_log.csv"
    with open(path, encoding="utf-8") as f:
        rows = [(int(r["step"]), float(r["loss"])) for r in csv.DictReader(f)]
    assert len(rows) >= 2
    assert rows == sorted(rows), "шаги в train_log.csv не по порядку"
    k = max(1, len(rows) // 5)
    first = sum(l for _, l in rows[:k]) / k
    last = sum(l for _, l in rows[-k:]) / k
    assert last < first, f"лосс не упал: первые {first:.3f}, последние {last:.3f}"
