from __future__ import annotations

import io

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.detector import AnalysisResult, Detection


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def detections_to_dataframe(detections: list[Detection]) -> pd.DataFrame:
    if not detections:
        return pd.DataFrame(
            columns=[
                "Класс (EN)",
                "Класс (RU)",
                "Уверенность",
                "X1",
                "Y1",
                "X2",
                "Y2",
                "Площадь (px²)",
            ]
        )

    rows = []
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        rows.append(
            {
                "Класс (EN)": d.class_name,
                "Класс (RU)": d.class_name_ru,
                "Уверенность": round(d.confidence, 4),
                "X1": x1,
                "Y1": y1,
                "X2": x2,
                "Y2": y2,
                "Площадь (px²)": d.area,
            }
        )
    return pd.DataFrame(rows)


def results_summary_dataframe(results: list[AnalysisResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        classes = ", ".join(d.class_name_ru for d in r.detections) or "—"
        rows.append(
            {
                "Файл": r.source_name,
                "Вердикт": "Брак" if r.has_defects else "OK",
                "Дефектов": r.defect_count,
                "Типы дефектов": classes,
            }
        )
    return pd.DataFrame(rows)


def plot_class_distribution(results: list[AnalysisResult]) -> bytes | None:
    counts: dict[str, int] = {}
    for result in results:
        for det in result.detections:
            counts[det.class_name_ru] = counts.get(det.class_name_ru, 0) + 1

    if not counts:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = list(counts.keys())
    values = list(counts.values())
    ax.barh(labels, values, color="#e74c3c")
    ax.set_xlabel("Количество")
    ax.set_title("Распределение типов дефектов")
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()
