"""Запись для Вскрытия 5: держит ли модель требование, поставленное на первом шаге, через много шагов.

    python record_steps.py
    python record_steps.py --model Qwen/Qwen3-4B --steps 12 --series 8

Одна и та же работа в двух режимах. В режиме dialog это один длинный диалог: требования названы один раз
в системной инструкции, дальше идут вопросы, на каждый модель вызывает инструмент search, и все вызовы,
результаты и ответы копятся в контексте. В режиме fresh каждый вопрос идет в новом контексте, который
собирает код, и требования стоят в нем заново. Серия это случайная выборка вопросов в случайном порядке;
серий несколько, чтобы на каждый номер шага приходилась доля, а не один исход.

Требование на шаге считается выполненным, если модель вызвала поиск, ответила разборчивой строкой JSON
и сослалась на номер фрагмента из последнего вызова либо отказалась с пустым source.
С флагом --search by-code поиск делает код, а не модель: фрагменты приходят в реплике пользователя.

Формальное требование это еще не всё: модель может держать формат и при этом отвечать не по фрагментам
последнего вызова. Поэтому рядом считаются исходы по эталонам, как в лабораторной 5: доля выдуманных ответов
и доля отказов на вопросы без ответа в корпусе, по блокам из четырех шагов.

Пишет в fallback/: 06_steps.csv (все шаги), 06_steps.md (таблицы), 06_steps.png (график).
С флагом --replot таблицы и график перестраиваются по готовому csv, модель не загружается.
"""
import argparse
import csv
import json
import os
import random
import re
import time
import urllib.request
from pathlib import Path

import matplotlib
import numpy
import torch
from transformers import AutoModel
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

HERE_PATH = Path(__file__).resolve().parent
DATA_PATH = HERE_PATH.parents[1] / "assignments" / "05" / "data"
DATA_URL = "https://raw.githubusercontent.com/pmenshih/tgo-course/main/assignments/05/data/"
MODEL = "Qwen/Qwen3-0.6B"
ENCODER = "intfloat/multilingual-e5-small"

CHUNK_WORD_LIMIT = 120
MAX_NEW_TOKENS = 64
REFUSAL = "нет данных"
JSON_PATTERN = re.compile(r"\{.*?\}", re.DOTALL)
TOOL_CALL_TAG = "<tool_call>"
MODES = ["dialog", "fresh"]
BLOCK_STEPS = 4
INTEGER_FIELDS = [
    "series", "step", "question_id", "searched", "valid", "source_ok", "requirement_met", "context_tokens",
]

COURSE = "«Технологии глубокого обучения и нейронных сетей»"
REQUIREMENTS_BY_MODEL = (
    f"Ты отвечаешь на вопросы о курсе {COURSE}. Требования действуют на весь диалог. "
    "Перед каждым ответом вызови инструмент search с запросом по вопросу. "
    "Отвечай только по фрагментам, которые вернул последний вызов. "
    'Ответ это одна строка JSON вида {"answer": "...", "source": номер фрагмента из последнего вызова}. '
    'Если в этих фрагментах нет ответа, напиши {"answer": "нет данных", "source": null}.'
)
REQUIREMENTS_BY_CODE = (
    f"Ты отвечаешь на вопросы о курсе {COURSE}. Требования действуют на весь диалог. "
    "Отвечай только по фрагментам из последнего сообщения пользователя. "
    'Ответ это одна строка JSON вида {"answer": "...", "source": номер фрагмента из последнего сообщения}. '
    'Если в этих фрагментах нет ответа, напиши {"answer": "нет данных", "source": null}.'
)
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Найти фрагменты материалов курса по запросу",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "поисковый запрос"},
            },
            "required": ["query"],
        },
    },
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def system(text: str) -> dict:
    return {"role": "system", "content": text}


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


# ----------------------------------------------------------------------------- данные и поиск

def load_json(name: str):
    """Файл из репозитория курса; если его рядом нет, он скачивается."""
    path = DATA_PATH / name
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        with urllib.request.urlopen(DATA_URL + name) as response:
            text = response.read().decode("utf-8")
    return json.loads(text)


def split_document(document: dict) -> list[dict]:
    """Блоки по пустым строкам склеиваются под одним заголовком, пока в фрагменте не больше 120 слов."""
    chunks = []
    heading = ""
    blocks = []

    def close_chunk() -> None:
        if blocks:
            chunk = {
                "doc": document["doc"],
                "heading": heading,
                "text": "\n\n".join(blocks),
            }
            chunks.append(chunk)
            blocks.clear()

    for block in document["text"].split("\n\n"):
        block = block.strip()
        if not block:
            continue

        if block.startswith("#"):
            close_chunk()
            first_line, _, rest = block.partition("\n")
            heading = first_line.lstrip("#").strip()
            block = rest.strip()
            if not block:
                continue

        words_so_far = sum(len(part.split()) for part in blocks)
        if blocks and words_so_far + len(block.split()) > CHUNK_WORD_LIMIT:
            close_chunk()
        blocks.append(block)

    close_chunk()
    return chunks


def passage(chunk: dict) -> str:
    if chunk["heading"]:
        return chunk["heading"] + "\n" + chunk["text"]
    return chunk["text"]


class Workbench:
    """Модель, энкодер и индекс фрагментов: все, что нужно одному шагу."""

    def __init__(self, model_id: str, encoder_id: str, k: int, search_by_model: bool) -> None:
        self.device = pick_device()
        self.k = k
        self.search_by_model = search_by_model
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_id)
        self.encoder = AutoModel.from_pretrained(encoder_id).to(self.device).eval()

        self.chunks = []
        for document in load_json("corpus.json")["documents"]:
            self.chunks.extend(split_document(document))
        self.chunk_vectors = self.embed([passage(chunk) for chunk in self.chunks], "passage: ")

    @torch.no_grad()
    def embed(self, texts: list[str], prefix: str) -> numpy.ndarray:
        vectors = []
        for start in range(0, len(texts), 16):
            batch = [prefix + text for text in texts[start:start + 16]]
            inputs = self.encoder_tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            hidden = self.encoder(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            mean = torch.nn.functional.normalize(mean, dim=-1)
            vectors.append(mean.float().cpu().numpy())
        return numpy.concatenate(vectors)

    def search(self, query: str) -> str:
        """Текст k ближайших фрагментов с номерами от 1."""
        query_vector = self.embed([query], "query: ")[0]
        top = numpy.argsort(-(self.chunk_vectors @ query_vector))[:self.k]
        lines = []
        for number, chunk_id in enumerate(top, start=1):
            lines.append(f"[{number}] {passage(self.chunks[int(chunk_id)])}")
        return "Фрагменты:\n" + "\n\n".join(lines)

    @torch.no_grad()
    def generate(self, messages: list[dict]) -> tuple[str, int]:
        """Жадный ответ и длина промпта в токенах; в режиме поиска моделью в шаблон передается инструмент."""
        tools = [SEARCH_TOOL] if self.search_by_model else None
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_tokens = inputs["input_ids"].shape[1]
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        raw = self.tokenizer.decode(output_ids[0, prompt_tokens:], skip_special_tokens=True).strip()
        return raw, prompt_tokens


# ----------------------------------------------------------------------------- разбор ответов

def parse_tool_call(raw: str) -> dict | None:
    """Вызов инструмента из текста модели: JSON после тега <tool_call> с полями name и arguments."""
    position = raw.find(TOOL_CALL_TAG)
    if position < 0:
        return None
    start = raw.find("{", position)
    if start < 0:
        return None
    try:
        call, _ = json.JSONDecoder().raw_decode(raw[start:])
    except json.JSONDecodeError:
        return None
    else:
        if not isinstance(call, dict) or call.get("name") != "search":
            return None
        if not isinstance(call.get("arguments"), dict):
            return None
        return call


def parse_answer(raw: str) -> tuple[str | None, int | None]:
    """Поля answer и source из JSON-строки ответа; (None, None), если разобрать не удалось."""
    match = JSON_PATTERN.search(raw)
    if match is None:
        return None, None
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return None, None
    else:
        answer = parsed.get("answer")
        source = parsed.get("source")
        if not isinstance(answer, (str, int, float)) or isinstance(answer, bool) or not str(answer).strip():
            return None, None
        if not isinstance(source, int) or isinstance(source, bool):
            source = None
        return str(answer).strip(), source


def normalize(text: str) -> str:
    lowered = text.lower().replace("ё", "е")
    return "".join(symbol for symbol in lowered if symbol.isalnum())


def is_refusal(answer: str) -> bool:
    return normalize(REFUSAL) in normalize(answer)


def outcome(question: dict, answer: str | None) -> str:
    if answer is not None and is_refusal(answer):
        return "refusal_correct" if not question["answerable"] else "refusal_wrong"
    normalized = normalize(answer or "")
    if question["answerable"] and any(normalize(reference) in normalized for reference in question["answers"]):
        return "correct"
    return "fabricated"


# ----------------------------------------------------------------------------- шаг и серии

def run_step(bench: Workbench, messages: list[dict], question: dict) -> dict:
    """Один вопрос в контексте messages; контекст дополняется репликами этого шага."""
    searched = False
    if bench.search_by_model:
        messages.append(user(f"Вопрос: {question['question']}"))
        raw, prompt_tokens = bench.generate(messages)
        call = parse_tool_call(raw)
        if call is not None:
            searched = True
            query = str(call["arguments"].get("query") or question["question"])
            tool_call = {"type": "function", "function": {"name": "search", "arguments": call["arguments"]}}
            messages.append({"role": "assistant", "content": "", "tool_calls": [tool_call]})
            messages.append({"role": "tool", "content": bench.search(query)})
            raw, prompt_tokens = bench.generate(messages)
    else:
        searched = True
        messages.append(user(f"{bench.search(question['question'])}\n\nВопрос: {question['question']}"))
        raw, prompt_tokens = bench.generate(messages)
    messages.append(assistant(raw))

    answer, source = parse_answer(raw)
    valid = answer is not None
    refused = valid and is_refusal(answer)
    source_ok = (refused and source is None) or (not refused and source is not None and 1 <= source <= bench.k)
    return {
        "question_id": question["id"],
        "searched": int(searched),
        "valid": int(valid),
        "source_ok": int(valid and source_ok),
        "requirement_met": int(searched and valid and source_ok),
        "outcome": outcome(question, answer),
        "context_tokens": prompt_tokens,
        "raw": raw,
    }


def run_series(bench: Workbench, questions: list[dict], requirements: str, fresh_rows: dict) -> list[dict]:
    """Серия вопросов одним диалогом и те же вопросы по одному в новом контексте."""
    rows = []
    dialog = [system(requirements)]
    for step, question in enumerate(questions, start=1):
        dialog_row = run_step(bench, dialog, question)
        dialog_row.update({"mode": "dialog", "step": step})
        rows.append(dialog_row)

        if question["id"] not in fresh_rows:
            fresh_rows[question["id"]] = run_step(bench, [system(requirements)], question)
        fresh_row = dict(fresh_rows[question["id"]])
        fresh_row.update({"mode": "fresh", "step": step})
        rows.append(fresh_row)
    return rows


def summarize(rows: list[dict], steps: int) -> list[dict]:
    summary = []
    for step in range(1, steps + 1):
        line = {"step": step}
        for mode in MODES:
            step_rows = [row for row in rows if row["mode"] == mode and row["step"] == step]
            count = len(step_rows)
            line[f"{mode}_met"] = sum(row["requirement_met"] for row in step_rows) / count
            line[f"{mode}_searched"] = sum(row["searched"] for row in step_rows) / count
            line[f"{mode}_correct"] = sum(row["outcome"] == "correct" for row in step_rows) / count
            line[f"{mode}_tokens"] = sum(row["context_tokens"] for row in step_rows) / count
        summary.append(line)
    return summary


def summarize_blocks(rows: list[dict], steps: int, unanswerable_ids: set) -> list[dict]:
    """Исходы по блокам шагов: на одном номере шага примеров мало, на блоке из четырех шагов вчетверо больше."""
    blocks = []
    for first in range(1, steps + 1, BLOCK_STEPS):
        last = min(first + BLOCK_STEPS - 1, steps)
        line = {"first": first, "last": last}
        for mode in MODES:
            block_rows = [row for row in rows if row["mode"] == mode and first <= row["step"] <= last]
            unanswerable = [row for row in block_rows if row["question_id"] in unanswerable_ids]
            line[f"{mode}_answers"] = len(block_rows)
            line[f"{mode}_fabricated"] = sum(row["outcome"] == "fabricated" for row in block_rows)
            line[f"{mode}_correct"] = sum(row["outcome"] == "correct" for row in block_rows)
            line[f"{mode}_unanswerable"] = len(unanswerable)
            line[f"{mode}_refused"] = sum(row["outcome"] == "refusal_correct" for row in unanswerable)
            line[f"{mode}_not_searched"] = sum(1 - row["searched"] for row in block_rows)
        blocks.append(line)
    return blocks


def write_table(path: Path, summary: list[dict], blocks: list[dict], title: str) -> None:
    lines = [
        title,
        "",
        "| Шаг | Требование, диалог | Требование, новый контекст | Поиск вызван, диалог | Верных, диалог "
        "| Верных, новый контекст | Токенов, диалог | Токенов, новый контекст |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for line in summary:
        lines.append(
            f"| {line['step']} | {line['dialog_met']:.2f} | {line['fresh_met']:.2f} "
            f"| {line['dialog_searched']:.2f} | {line['dialog_correct']:.2f} | {line['fresh_correct']:.2f} "
            f"| {line['dialog_tokens']:.0f} | {line['fresh_tokens']:.0f} |"
        )
    lines.extend(
        [
            "",
            "Исходы по блокам шагов, число ответов:",
            "",
            "| Шаги | Режим | Ответов | Верных | Выдуманных | Вопросов без ответа | Из них отказ | Поиск не вызван |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    names = {"dialog": "один диалог", "fresh": "новый контекст"}
    for block in blocks:
        for mode in MODES:
            lines.append(
                f"| {block['first']}–{block['last']} | {names[mode]} | {block[f'{mode}_answers']} "
                f"| {block[f'{mode}_correct']} | {block[f'{mode}_fabricated']} | {block[f'{mode}_unanswerable']} "
                f"| {block[f'{mode}_refused']} | {block[f'{mode}_not_searched']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path: Path, summary: list[dict], blocks: list[dict], title: str) -> None:
    """Слева формальное требование по номеру шага, справа доля выдуманных ответов по блокам шагов."""
    steps = [line["step"] for line in summary]
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4))

    left.plot(steps, [line["dialog_met"] for line in summary], color="#1A1A1A", marker="o", label="один длинный диалог")
    left.plot(
        steps,
        [line["fresh_met"] for line in summary],
        color="#888888",
        marker="o",
        linestyle="--",
        label="новый контекст на каждый вопрос",
    )
    left.set_ylim(-0.05, 1.05)
    left.set_xticks(steps)
    left.set_xlabel("номер шага")
    left.set_title("поиск вызван, JSON, ссылка на фрагмент")

    positions = numpy.arange(len(blocks))
    width = 0.38
    dialog_share = [block["dialog_fabricated"] / block["dialog_answers"] for block in blocks]
    fresh_share = [block["fresh_fabricated"] / block["fresh_answers"] for block in blocks]
    right.bar(positions - width / 2, dialog_share, width, color="#1A1A1A", label="один длинный диалог")
    right.bar(positions + width / 2, fresh_share, width, color="#BBBBBB", label="новый контекст на каждый вопрос")
    right.set_xticks(positions)
    right.set_xticklabels([f"шаги {block['first']}–{block['last']}" for block in blocks])
    right.set_ylim(0, 1.05)
    right.set_title("доля выдуманных ответов")

    for axis in (left, right):
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = right.get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, loc="lower center", ncol=2)
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0.08, 1, 0.95))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def read_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for field in INTEGER_FIELDS:
            row[field] = int(row[field])
    return rows


def write_results(output_path: Path, name: str, rows: list[dict], steps: int, title: str) -> None:
    unanswerable_ids = {question["id"] for question in load_json("questions.json") if not question["answerable"]}
    summary = summarize(rows, steps)
    blocks = summarize_blocks(rows, steps, unanswerable_ids)
    write_table(output_path / f"{name}.md", summary, blocks, title)
    write_plot(output_path / f"{name}.png", summary, blocks, title)

    for line in summary:
        log(
            f"шаг {line['step']:2d}: требование {line['dialog_met']:.2f} в диалоге и {line['fresh_met']:.2f} "
            f"в новом контексте, токенов {line['dialog_tokens']:.0f} и {line['fresh_tokens']:.0f}"
        )
    for block in blocks:
        log(
            f"шаги {block['first']}–{block['last']}: выдуманных {block['dialog_fabricated']} в диалоге "
            f"и {block['fresh_fabricated']} в новом контексте из {block['dialog_answers']}; "
            f"отказов на вопросы без ответа {block['dialog_refused']} и {block['fresh_refused']} "
            f"из {block['dialog_unanswerable']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("LAB_CHAT_MODEL", MODEL))
    parser.add_argument("--encoder", default=os.environ.get("LAB_ENCODER", ENCODER))
    parser.add_argument("--steps", type=int, default=12, help="вопросов в серии")
    parser.add_argument("--series", type=int, default=8, help="сколько серий")
    parser.add_argument("--k", type=int, default=3, help="фрагментов на один поиск")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--search", choices=["by-model", "by-code"], default="by-model")
    parser.add_argument("--out", default=str(HERE_PATH / "fallback"))
    parser.add_argument("--name", default="06_steps", help="общее имя трех файлов результата")
    parser.add_argument("--replot", action="store_true", help="перестроить таблицы и график по готовому csv")
    arguments = parser.parse_args()

    output_path = Path(arguments.out)
    output_path.mkdir(parents=True, exist_ok=True)
    if arguments.replot:
        rows = read_rows(output_path / f"{arguments.name}.csv")
        steps = max(row["step"] for row in rows)
        series = max(row["series"] for row in rows)
        title = f"{arguments.model}, серий {series}, шагов {steps}"
        write_results(output_path, arguments.name, rows, steps, title)
        return

    search_by_model = arguments.search == "by-model"
    requirements = REQUIREMENTS_BY_MODEL if search_by_model else REQUIREMENTS_BY_CODE

    bench = Workbench(arguments.model, arguments.encoder, arguments.k, search_by_model)
    questions = load_json("questions.json")
    log(
        f"устройство {bench.device}, модель {arguments.model}, поиск {arguments.search}, "
        f"фрагментов {len(bench.chunks)}"
    )

    generator = random.Random(arguments.seed)
    rows = []
    fresh_rows = {}
    for series in range(1, arguments.series + 1):
        sample = generator.sample(questions, arguments.steps)
        series_rows = run_series(bench, sample, requirements, fresh_rows)
        for row in series_rows:
            row["series"] = series
        rows.extend(series_rows)

        dialog_rows = [row for row in series_rows if row["mode"] == "dialog"]
        marks = "".join(str(row["requirement_met"]) for row in dialog_rows)
        last_tokens = dialog_rows[-1]["context_tokens"]
        log(f"серия {series}: требование по шагам {marks}, контекст на последнем шаге {last_tokens}")

    fields = [
        "series", "mode", "step", "question_id", "searched", "valid", "source_ok", "requirement_met",
        "outcome", "context_tokens", "raw",
    ]
    with open(output_path / f"{arguments.name}.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    title = f"{arguments.model}, серий {arguments.series}, шагов {arguments.steps}"
    write_results(output_path, arguments.name, rows, arguments.steps, title)
    log("готово")


if __name__ == "__main__":
    main()
