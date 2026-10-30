"""Работа 1 заочного отделения. Задание: part-time/assignments/01/README.md в репозитории курса.

Запуск: python pt01/run.py, для CI python pt01/run.py --fast.
Артефакты bigram.json и bpe.json пишутся в pt01/artifacts/.
"""
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
CORPUS = HERE / "corpus.txt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="быстрый режим для CI")
    args = ap.parse_args()
    ARTIFACTS.mkdir(exist_ok=True)
    raise NotImplementedError("работа не начата")


if __name__ == "__main__":
    main()
