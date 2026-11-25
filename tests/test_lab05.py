"""Тесты лабораторной 5: читают lab05/artifacts/ и пересчитывают из данных все, что можно пересчитать.

Скрипт студента тесты не запускают и не импортируют. Корпус и вопросы с эталонными ответами
берутся из tests/data/lab05/, это копии файлов из assignments/05/data/.
"""
import csv
import json
import re
from pathlib import Path

import pytest

ROOT_PATH = Path(__file__).resolve().parents[1]
LAB_PATH = ROOT_PATH / "lab05"
ARTIFACTS_PATH = LAB_PATH / "artifacts"
DATA_PATH = Path(__file__).resolve().parent / "data" / "lab05"

MODES = ["no_search", "rag", "rag_verified"]
SHARES = ["correct", "fabricated", "refusal_correct", "refusal_wrong"]
COLUMNS = [
    "id", "answerable", "no_search", "rag", "rag_source", "verdict", "rag_verified",
    "raw_no_search", "raw_rag", "raw_verifier",
]
REFUSAL = "нет данных"
VERDICT_YES = "да"
VERDICT_NO = "нет"
TOP_K = 5
MIN_CHUNKS = 100
MAX_CHUNK_WORDS = 300
MIN_QUESTIONS = 10
TOLERANCE = 2e-3
JSON_PATTERN = re.compile(r"\{.*?\}", re.DOTALL)

pytestmark = pytest.mark.skipif(
    "NotImplementedError" in (LAB_PATH / "run.py").read_text(encoding="utf-8"),
    reason="lab05/run.py это заглушка, работа не начата",
)


def load_json(path: Path):
    assert path.exists(), f"нет файла {path.relative_to(ROOT_PATH)}"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def metrics():
    return load_json(ARTIFACTS_PATH / "metrics.json")


@pytest.fixture(scope="module")
def chunks():
    return load_json(ARTIFACTS_PATH / "chunks.json")


@pytest.fixture(scope="module")
def retrieval():
    return load_json(ARTIFACTS_PATH / "retrieval.json")


@pytest.fixture(scope="module")
def thresholds():
    return load_json(DATA_PATH / "thresholds.json")


@pytest.fixture(scope="module")
def documents():
    corpus = load_json(DATA_PATH / "corpus.json")
    return {document["doc"]: document["text"] for document in corpus["documents"]}


@pytest.fixture(scope="module")
def questions():
    return {question["id"]: question for question in load_json(DATA_PATH / "questions.json")}


@pytest.fixture(scope="module")
def answers():
    path = ARTIFACTS_PATH / "answers.csv"
    assert path.exists(), "нет lab05/artifacts/answers.csv, запустите python lab05/run.py --fast"
    with open(path, encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows, "answers.csv пуст"
    return rows


def normalize(text: str) -> str:
    lowered = text.lower().replace("ё", "е")
    return "".join(symbol for symbol in lowered if symbol.isalnum())


def squeeze(text: str) -> str:
    return "".join(text.split())


def is_refusal(answer: str) -> bool:
    return normalize(REFUSAL) in normalize(answer)


def parse_answer(raw: str) -> str:
    """Поле answer из первой JSON-строки сырого ответа; пустая строка, если разобрать не удалось."""
    match = JSON_PATTERN.search(raw)
    if match is None:
        return ""
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return ""
    else:
        answer = parsed.get("answer")
        if not isinstance(answer, (str, int, float)) or isinstance(answer, bool):
            return ""
        return str(answer).strip()


def same_answer(first: str, second: str) -> bool:
    first_normalized = normalize(first)
    second_normalized = normalize(second)
    if not first_normalized or not second_normalized:
        return False
    return first_normalized in second_normalized or second_normalized in first_normalized


def outcome(question: dict, answer: str) -> str:
    """Исход по правилам задания; пустая ячейка это ответ, который не разобрался, он считается выдуманным."""
    if answer.strip() and is_refusal(answer):
        return "refusal_correct" if not question["answerable"] else "refusal_wrong"
    normalized = normalize(answer)
    matches = [reference for reference in question["answers"] if normalize(reference) in normalized]
    if answer.strip() and question["answerable"] and matches:
        return "correct"
    return "fabricated"


def recompute_mode(rows: list[dict], questions: dict, mode: str) -> dict:
    outcomes = [outcome(questions[int(row["id"])], row[mode]) for row in rows]
    answerable = sum(1 for row in rows if questions[int(row["id"])]["answerable"])
    unanswerable = len(rows) - answerable
    return {
        "correct": outcomes.count("correct") / answerable,
        "fabricated": outcomes.count("fabricated") / len(rows),
        "refusal_correct": outcomes.count("refusal_correct") / unanswerable,
        "refusal_wrong": outcomes.count("refusal_wrong") / answerable,
        "invalid": sum(1 for row in rows if not row[mode].strip()),
    }


def threshold_key(metrics: dict, name: str) -> str:
    return f"{name}_fast" if metrics.get("fast") else f"{name}_full"


# ---------------------------------------------------------------- chunks.json

def test_chunks(chunks, documents, metrics):
    assert len(chunks) >= MIN_CHUNKS, f"фрагментов {len(chunks)}, по базовому правилу их около двухсот"
    assert metrics["n_chunks"] == len(chunks), "n_chunks в metrics.json не совпадает с chunks.json"
    assert [chunk["id"] for chunk in chunks] == list(range(len(chunks))), "id фрагментов идут не подряд с нуля"

    squeezed_documents = {name: squeeze(text) for name, text in documents.items()}
    for chunk in chunks:
        assert chunk["doc"] in squeezed_documents, f"фрагмент {chunk['id']}: в корпусе нет документа {chunk['doc']!r}"
        assert chunk["text"].strip(), f"фрагмент {chunk['id']} пустой"
        assert squeeze(chunk["text"]) in squeezed_documents[chunk["doc"]], (
            f"текст фрагмента {chunk['id']} не найден в документе {chunk['doc']}"
        )
        words = len(chunk["text"].split())
        assert words <= MAX_CHUNK_WORDS, f"фрагмент {chunk['id']}: {words} слов, предел {MAX_CHUNK_WORDS}"


# ---------------------------------------------------------------- retrieval.json

def test_retrieval_schema(retrieval, chunks, answers):
    assert retrieval["k"] == TOP_K, f"k = {retrieval['k']}, в задании {TOP_K}"
    ids = [row["id"] for row in retrieval["questions"]]
    assert ids == [int(row["id"]) for row in answers], "вопросы в retrieval.json и answers.csv не совпадают"
    for row in retrieval["questions"]:
        assert len(row["top"]) == TOP_K, f"вопрос {row['id']}: найдено {len(row['top'])} фрагментов, нужно {TOP_K}"
        assert len(set(row["top"])) == TOP_K, f"вопрос {row['id']}: номера фрагментов повторяются"
        assert all(0 <= chunk_id < len(chunks) for chunk_id in row["top"]), f"вопрос {row['id']}: номер вне chunks.json"
        assert len(row["scores"]) == TOP_K
        assert row["scores"] == sorted(row["scores"], reverse=True), f"вопрос {row['id']}: близости не по убыванию"


def test_retrieval_hits(retrieval, chunks, questions, metrics, thresholds):
    hits = []
    for row in retrieval["questions"]:
        question = questions[row["id"]]
        if not question["answerable"]:
            assert row["hit"] is None, f"вопрос {row['id']} без ответа, hit должен быть null"
            continue
        evidence = squeeze(question["evidence"])
        hit = any(evidence in squeeze(chunks[chunk_id]["text"]) for chunk_id in row["top"])
        assert row["hit"] == hit, f"вопрос {row['id']}: hit в файле {row['hit']}, по фрагментам {hit}"
        hits.append(hit)

    rate = sum(hits) / len(hits)
    assert retrieval["hit_rate"] == pytest.approx(rate, abs=TOLERANCE)
    key = threshold_key(metrics, "hit_rate")
    assert rate >= thresholds[key], f"доля попаданий поиска {rate:.2f}, порог {thresholds[key]}"


# ---------------------------------------------------------------- answers.csv и metrics.json

def test_answers_schema(answers, questions, metrics):
    for column in COLUMNS:
        assert column in answers[0], f"в answers.csv нет колонки {column}"
    assert len(answers) >= MIN_QUESTIONS, f"в answers.csv меньше {MIN_QUESTIONS} строк"
    assert metrics["n_questions"] == len(answers), "n_questions в metrics.json не совпадает с answers.csv"

    expected_ids = sorted(questions)[:len(answers)]
    assert [int(row["id"]) for row in answers] == expected_ids, "вопросы идут не в порядке набора с первого"

    unanswerable = 0
    for row in answers:
        question = questions[int(row["id"])]
        assert int(row["answerable"]) == int(question["answerable"]), f"вопрос {row['id']}: answerable не как в наборе"
        assert row["verdict"] in ("", VERDICT_YES, VERDICT_NO), f"вопрос {row['id']}: вердикт {row['verdict']!r}"
        unanswerable += not question["answerable"]
    assert metrics["n_unanswerable"] == unanswerable


@pytest.mark.parametrize("mode", MODES)
def test_metrics_recomputed(metrics, answers, questions, mode):
    assert mode in metrics["modes"], f"в metrics.json нет прогона {mode}"
    written = metrics["modes"][mode]
    recomputed = recompute_mode(answers, questions, mode)
    for share in SHARES:
        assert written[share] == pytest.approx(recomputed[share], abs=TOLERANCE), (
            f"{mode}: {share} в metrics.json {written[share]}, по answers.csv {recomputed[share]:.4f}"
        )
    assert written["invalid"] == recomputed["invalid"]
    assert written["prompt_tokens"] > 0
    assert written["sec_per_question"] > 0


def test_verified_follows_verdicts(answers):
    for row in answers:
        checked = bool(row["rag"].strip()) and not is_refusal(row["rag"])
        if not checked:
            assert row["verdict"] == "", f"вопрос {row['id']}: отказ или неразобранный ответ не проверяется"
            assert row["rag_verified"] == row["rag"]
            continue
        assert row["verdict"] in (VERDICT_YES, VERDICT_NO), f"вопрос {row['id']}: ответ не проверен"
        if row["verdict"] == VERDICT_NO:
            assert is_refusal(row["rag_verified"]), f"вопрос {row['id']}: помеченный ответ не заменен отказом"
        else:
            assert row["rag_verified"] == row["rag"], f"вопрос {row['id']}: ответ изменен при вердикте «да»"


def test_verdicts_recomputed(answers):
    """Вердикт решает код: «да», если повторный ответ разобрался, не отказ и совпал с первым."""
    for row in answers:
        if not row["verdict"]:
            continue
        second = parse_answer(row["raw_verifier"])
        agreed = bool(second) and not is_refusal(second) and same_answer(row["rag"], second)
        expected = VERDICT_YES if agreed else VERDICT_NO
        assert row["verdict"] == expected, (
            f"вопрос {row['id']}: вердикт {row['verdict']!r}, "
            f"по повторному ответу {row['raw_verifier']!r} выходит {expected!r}"
        )


def test_verifier_recomputed(metrics, answers, questions):
    flagged = 0
    flagged_fabricated = 0
    fabricated = 0
    for row in answers:
        if not row["verdict"]:
            continue
        is_fabricated = outcome(questions[int(row["id"])], row["rag"]) == "fabricated"
        fabricated += is_fabricated
        if row["verdict"] == VERDICT_NO:
            flagged += 1
            flagged_fabricated += is_fabricated

    written = metrics["verifier"]
    assert written["flagged"] == flagged, f"flagged в metrics.json {written['flagged']}, по answers.csv {flagged}"
    if flagged:
        assert written["precision"] == pytest.approx(flagged_fabricated / flagged, abs=TOLERANCE)
    if fabricated:
        assert written["recall"] == pytest.approx(flagged_fabricated / fabricated, abs=TOLERANCE)


# ---------------------------------------------------------------- качество

def test_search_helps(metrics, thresholds):
    gain = metrics["modes"]["rag"]["correct"] - metrics["modes"]["no_search"]["correct"]
    key = threshold_key(metrics, "correct_gain")
    assert gain >= thresholds[key], f"поиск поднял долю верных ответов на {gain:.2f}, порог {thresholds[key]}"


def test_modes_differ(answers):
    no_search = [row["no_search"] for row in answers]
    rag = [row["rag"] for row in answers]
    assert no_search != rag, "прогоны без поиска и с поиском дали одинаковые ответы на все вопросы"
    assert len(set(rag)) >= 2, "прогон с поиском отвечает одно и то же на все вопросы"
