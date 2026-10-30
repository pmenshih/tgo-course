"""Тесты работы 1 заочного отделения: читают pt01/artifacts/ и пересчитывают из данных все, что можно пересчитать.

Скрипт студента тесты не запускают и не импортируют, артефакты к их запуску уже должны лежать на месте.
Обязательные артефакты: bigram.json и bpe.json.
"""
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "pt01"
ART = LAB / "artifacts"
DATA = Path(__file__).resolve().parent / "data" / "pt01"

pytestmark = pytest.mark.skipif(
    "NotImplementedError" in (LAB / "run.py").read_text(encoding="utf-8"),
    reason="pt01/run.py это заглушка, работа не начата"
)


def load_json(path: Path):
    assert path.exists(), f"нет файла {path.relative_to(ROOT)}, запустите python pt01/run.py --fast"
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- bigram.json

@pytest.fixture(scope="module")
def bigram():
    return load_json(ART / "bigram.json")


def test_split(bigram):
    assert bigram["train_words"] + bigram["val_words"] == bigram["words"]
    frac = bigram["val_words"] / bigram["words"]
    assert 0.08 <= frac <= 0.12, f"отложено {frac:.1%} слов, должно быть около 10 %"
    assert bigram["words"] > 300_000, "корпус меньше ожидаемого: это точно «Война и мир» целиком?"


def test_top20(bigram):
    top = bigram["top20"]
    assert len(top) == 20
    counts = [c for _, c in top]
    assert counts == sorted(counts, reverse=True), "top20 должен быть отсортирован по убыванию"
    ref = set(json.loads((DATA / "top20.json").read_text(encoding="utf-8")))
    got = {w for w, _ in top}
    assert len(ref & got) >= 15, f"совпало только {len(ref & got)} слов из 20 с эталоном: проверьте регулярное выражение и регистр"


def test_perplexity(bigram):
    ppl = bigram["perplexity"]
    for a in ("1", "0.1", "0.01"):
        assert a in ppl, f"нет перплексии при alpha={a}"
        assert 1 < ppl[a] < 1e6 and math.isfinite(ppl[a])
    assert ppl["0.01"] < ppl["1"], "при alpha=1 сглаживание размазывает массу по словарю, перплексия должна быть выше"


def test_likelihood(bigram):
    lk = bigram["likelihood"]
    assert len(lk["sentence"].split()) >= 8
    assert sorted(lk["sentence"].split()) == sorted(lk["shuffled"].split()), "перемешанное предложение должно состоять из тех же слов"
    assert lk["logprob_sentence"] > lk["logprob_shuffled"], "связное предложение должно быть правдоподобнее перемешанного"
    assert lk["logprob_sentence"] < 0


def test_generation(bigram):
    gen = bigram["generation"]
    temps = {g["temperature"] for g in gen}
    assert {0.5, 1.0, 2.0} <= temps, "нужны образцы при температурах 0.5, 1 и 2"
    for g in gen:
        assert len(g["text"].split()) >= 20, "каждый образец не короче двадцати слов"
        assert g["text"].split()[0] == g["start"]


# ---------------------------------------------------------------- bpe.json

@pytest.fixture(scope="module")
def bpe():
    return load_json(ART / "bpe.json")


def apply_merges(word: str, ranks: dict) -> list[str]:
    """Слово в буквы плюс </w>, на каждом шаге сливается пара с наименьшим рангом."""
    sym = list(word) + ["</w>"]
    while len(sym) > 1:
        ranked = [(ranks[pair], i) for i, pair in enumerate(zip(sym, sym[1:])) if pair in ranks]
        if not ranked:
            break
        _, i = min(ranked)
        sym = sym[:i] + [sym[i] + sym[i + 1]] + sym[i + 2:]
    return sym


def test_bpe_merges_count(bpe):
    assert len(bpe["merges"]) == bpe["num_merges"]
    assert bpe["num_merges"] >= 200


def test_bpe_examples_reproduce(bpe):
    ranks = {}
    for i, (a, b) in enumerate(bpe["merges"]):
        ranks.setdefault((a, b), i)
    for word in ("наполеон", "сказала", "неудовольствие"):
        assert word in bpe["examples"], f"в examples нет слова «{word}»"
    for word, tokens in bpe["examples"].items():
        assert apply_merges(word, ranks) == tokens, f"слияния по порядку дают для «{word}» другое разбиение"


def test_bpe_stats_plausible(bpe):
    assert 1.2 <= bpe["stats"]["tokens_per_word"] <= 4.0
    assert 1.3 <= bpe["stats"]["compression_vs_chars"] <= 4.0
