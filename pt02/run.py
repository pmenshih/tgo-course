"""Работа 2 заочного отделения. Задание: part-time/assignments/02/README.md в репозитории курса.

Запуск: python pt02/run.py, для CI python pt02/run.py --fast.
Слияния BPE берутся из pt01/artifacts/bpe.json.
"""
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
BPE = HERE.parent / "pt01" / "artifacts" / "bpe.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="быстрый режим для CI")
    args = ap.parse_args()
    ARTIFACTS.mkdir(exist_ok=True)
    raise NotImplementedError("работа не начата")


if __name__ == "__main__":
    main()
