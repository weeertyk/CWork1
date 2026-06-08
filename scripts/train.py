#!/usr/bin/env python3
"""CLI для обучения модели детекции дефектов."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trainer import DefectTrainer, TrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение YOLOv8 для дефектов металла")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "config" / "neu_det.yaml",
        help="Путь к data.yaml (по умолчанию config/neu_det.yaml)",
    )
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--name", type=str, default="metal_defect_train")
    args = parser.parse_args()

    trainer = DefectTrainer()
    result = trainer.train(
        TrainingConfig(
            data_yaml=args.data,
            base_model=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            name=args.name,
        )
    )

    print(f"Лучшие веса: {result.best_weights}")
    print(f"Скопировано в models/best.pt")


if __name__ == "__main__":
    main()
