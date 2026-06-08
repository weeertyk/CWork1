from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO

from src.constants import DEFAULT_MODEL, MODELS_DIR, RUNS_DIR


@dataclass
class TrainingConfig:
    data_yaml: Path
    base_model: str = "yolov8n.pt"
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    device: str | None = None
    project: Path = RUNS_DIR / "detect"
    name: str = "metal_defect_train"


@dataclass
class TrainingResult:
    best_weights: Path
    last_weights: Path
    save_dir: Path
    metrics: dict


class DefectTrainer:
    """Обучение YOLOv8 на датасете дефектов металлических деталей."""

    def train(self, config: TrainingConfig) -> TrainingResult:
        if not config.data_yaml.exists():
            raise FileNotFoundError(f"data.yaml не найден: {config.data_yaml}")

        model = YOLO(config.base_model)
        results = model.train(
            data=str(config.data_yaml),
            epochs=config.epochs,
            imgsz=config.imgsz,
            batch=config.batch,
            device=config.device,
            project=str(config.project),
            name=config.name,
            exist_ok=True,
            pretrained=True,
            verbose=True,
        )

        save_dir = Path(results.save_dir)
        best_weights = save_dir / "weights" / "best.pt"
        last_weights = save_dir / "weights" / "last.pt"

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, DEFAULT_MODEL)

        metrics = {}
        if hasattr(results, "results_dict") and results.results_dict:
            metrics = dict(results.results_dict)

        return TrainingResult(
            best_weights=best_weights,
            last_weights=last_weights,
            save_dir=save_dir,
            metrics=metrics,
        )

    def validate(self, weights: Path, data_yaml: Path, device: str | None = None) -> dict:
        model = YOLO(str(weights))
        metrics = model.val(data=str(data_yaml), device=device, verbose=False)
        return {
            "mAP50": float(getattr(metrics.box, "map50", 0.0)),
            "mAP50-95": float(getattr(metrics.box, "map", 0.0)),
            "precision": float(getattr(metrics.box, "mp", 0.0)),
            "recall": float(getattr(metrics.box, "mr", 0.0)),
        }
