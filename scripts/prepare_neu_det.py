#!/usr/bin/env python3
"""Подготовка NEU-DET датасета для обучения YOLOv8."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_utils import prepare_neu_det_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Конвертация NEU-DET в формат YOLO")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Папка с исходным NEU-DET (IMAGES + ANNOTATIONS)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "neu_det_yolo",
        help="Куда сохранить подготовленный датасет",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    yaml_path = prepare_neu_det_pipeline(
        raw_dir=args.raw_dir,
        output_root=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    print(f"Готово. data.yaml: {yaml_path.resolve()}")
    print(f"config/neu_det.yaml обновлён.")


if __name__ == "__main__":
    main()
