"""Лабораторная 5. Задание: assignments/05/README.md в репозитории курса.

Запуск: python lab05/run.py, для CI python lab05/run.py --fast.
Корпус и вопросы скрипт скачивает сам в lab05/data/.
"""
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
DATA = HERE / "data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="быстрый режим для CI")
    args = ap.parse_args()
    ARTIFACTS.mkdir(exist_ok=True)
    raise NotImplementedError("работа не начата")


if __name__ == "__main__":
    main()
