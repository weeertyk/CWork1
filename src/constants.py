from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_MODEL = MODELS_DIR / "best.pt"

# NEU Steel Surface Defect Database — 6 типов дефектов
DEFECT_CLASSES = [
    "crazing",           # растрескивание
    "inclusion",         # включения
    "patches",           # пятна/участки
    "pitted_surface",    # точечная поверхность
    "rolled-in_scale",   # вrolled-in scale / окалина
    "scratches",         # царапины
]

DEFECT_CLASSES_RU = {
    "crazing": "Растрескивание",
    "inclusion": "Включения",
    "patches": "Пятна",
    "pitted_surface": "Точечная поверхность",
    "rolled-in_scale": "Окалина",
    "scratches": "Царапины",
}

YOLO_BASE_MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
