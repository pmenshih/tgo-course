"""Лабораторная 4. Задание: assignments/04/README.md в репозитории курса.

Запуск: python lab04/run.py, для CI python lab04/run.py --fast.
Адаптер обучается в lab04/train.ipynb и лежит в lab04/artifacts/adapter/.
"""
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
ADAPTER = ARTIFACTS / "adapter"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="быстрый режим для CI")
    args = ap.parse_args()
    ARTIFACTS.mkdir(exist_ok=True)
    raise NotImplementedError("работа не начата")


if __name__ == "__main__":
    main()
