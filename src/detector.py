from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from src.constants import DEFECT_CLASSES_RU, DEFAULT_MODEL, MODELS_DIR


@dataclass
class Detection:
    class_id: int
    class_name: str
    class_name_ru: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    area: int


@dataclass
class AnalysisResult:
    source_name: str
    image_bgr: np.ndarray
    annotated_bgr: np.ndarray
    detections: list[Detection] = field(default_factory=list)
    has_defects: bool = False
    verdict: str = "OK"

    @property
    def defect_count(self) -> int:
        return len(self.detections)


class DefectDetector:
    """Обёртка над YOLOv8 для детекции дефектов на металлических деталях."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        device: str | None = None,
    ) -> None:
        self.conf = conf
        self.iou = iou
        self.device = device
        self.model_path = self._resolve_model_path(model_path)
        self.model = YOLO(str(self.model_path))
        self.class_names: dict[int, str] = dict(self.model.names)

    @staticmethod
    def _resolve_model_path(model_path: str | Path | None) -> Path:
        if model_path:
            path = Path(model_path)
            if path.exists():
                return path
            raise FileNotFoundError(f"Модель не найдена: {path}")

        if DEFAULT_MODEL.exists():
            return DEFAULT_MODEL

        latest = DefectDetector.find_latest_trained_weights()
        if latest:
            return latest

        return Path("yolov8n.pt")

    @staticmethod
    def find_latest_trained_weights() -> Path | None:
        runs_root = Path("runs") / "detect"
        if not runs_root.exists():
            return None

        candidates = sorted(
            runs_root.glob("*/weights/best.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    def predict_image(
        self,
        image: np.ndarray | Image.Image | str | Path,
        source_name: str = "image",
    ) -> AnalysisResult:
        image_bgr = self._to_bgr(image)
        results = self.model.predict(
            source=image_bgr,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )[0]

        detections = self._parse_detections(results)
        annotated = results.plot()
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        has_defects = len(detections) > 0
        verdict = "DEFECT" if has_defects else "OK"

        return AnalysisResult(
            source_name=source_name,
            image_bgr=image_bgr,
            annotated_bgr=annotated_bgr,
            detections=detections,
            has_defects=has_defects,
            verdict=verdict,
        )

    def predict_batch(
        self,
        sources: list[np.ndarray | Image.Image | str | Path],
        source_names: list[str] | None = None,
    ) -> list[AnalysisResult]:
        names = source_names or [f"image_{i + 1}" for i in range(len(sources))]
        return [
            self.predict_image(src, name)
            for src, name in zip(sources, names, strict=True)
        ]

    def _parse_detections(self, results: Any) -> list[Detection]:
        detections: list[Detection] = []
        if results.boxes is None:
            return detections

        for box in results.boxes:
            class_id = int(box.cls.item())
            class_name = self.class_names.get(class_id, str(class_id))
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            area = max(0, x2 - x1) * max(0, y2 - y1)

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    class_name_ru=DEFECT_CLASSES_RU.get(class_name, class_name),
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    area=area,
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    @staticmethod
    def _to_bgr(image: np.ndarray | Image.Image | str | Path) -> np.ndarray:
        if isinstance(image, (str, Path)):
            bgr = cv2.imread(str(image))
            if bgr is None:
                raise ValueError(f"Не удалось прочитать изображение: {image}")
            return bgr

        if isinstance(image, Image.Image):
            rgb = np.array(image.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            if image.shape[2] == 3:
                # Streamlit/PIL обычно отдаёт RGB
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image

        raise TypeError(f"Неподдерживаемый тип изображения: {type(image)}")

    def is_pretrained_for_defects(self) -> bool:
        path = Path(self.model_path)
        if path.name == "yolov8n.pt" and not (MODELS_DIR / "best.pt").exists():
            if not DefectDetector.find_latest_trained_weights():
                return False
        return True
