# Metal Defect Analyzer

CV-приложение для детекции дефектов на металлических деталях с использованием **YOLOv8** и веб-интерфейса **Streamlit**.

Основано на подходе из [Kaggle notebook — Manufacturing Defect Detection using YOLOv8](https://www.kaggle.com/code/hareshkanaaramaraj/manufacturing-defect-detection-using-yolov8/notebook), адаптировано под датасет **NEU Steel Surface Defect Database** (дефекты поверхности металла/стали).

## Возможности

- **Анализ изображений** — загрузка файлов или съёмка с камеры, bounding boxes, отчёт CSV
- **Обучение модели** — fine-tuning YOLOv8 на NEU-DET или своём датасете
- **Подготовка данных** — конвертация XML-аннотаций NEU-DET в формат YOLO

### Типы дефектов (6 классов)

| Класс | Описание |
|-------|----------|
| crazing | Растрескивание |
| inclusion | Включения |
| patches | Пятна |
| pitted_surface | Точечная поверхность |
| rolled-in_scale | Окалина |
| scratches | Царапины |

## Быстрый старт

### 1. Установка

```bash
cd CWork1
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Скачать датасет

Зарегистрируйтесь на Kaggle и скачайте [NEU Surface Defect Database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database).

Альтернатива через Kaggle CLI:

```bash
pip install kaggle
kaggle datasets download -d kaustubhdikshit/neu-surface-defect-database
```

### 3. Подготовить данные

Положите распакованный датасет в `data/NEU-DET/`:

```
data/NEU-DET/
├── train/
│   ├── images/
│   │   ├── crazing/
│   │   ├── inclusion/
│   │   ├── patches/
│   │   ├── pitted_surface/
│   │   ├── rolled-in_scale/
│   │   └── scratches/
│   └── annotations/
└── validation/
    ├── images/...
    └── annotations/
```

Затем запустите:

```bash
python scripts/prepare_neu_det.py --raw-dir "data/NEU-DET"
```

Или через вкладку **«Подготовка данных»** в веб-интерфейсе.

### 4. Обучить модель

```bash
python scripts/train.py --data config/neu_det.yaml --epochs 50 --model yolov8n.pt
```

### 5. Запустить приложение

```bash
streamlit run app.py
```

Откроется браузер с интерфейсом на `http://localhost:8501`.

## Структура проекта

```
CWork1/
├── app.py                  # Streamlit UI
├── config/
│   └── neu_det.yaml        # Конфиг датасета YOLO
├── src/
│   ├── detector.py         # Inference YOLOv8
│   ├── trainer.py          # Обучение
│   ├── dataset_utils.py    # XML → YOLO, split
│   └── visualization.py    # Отчёты и графики
├── scripts/
│   ├── prepare_neu_det.py
│   └── train.py
├── models/                 # best.pt после обучения
└── data/                   # Подготовленный датасет
```

## Использование своей модели

После обучения веса сохраняются в:
- `runs/detect/metal_defect_train/weights/best.pt`
- `models/best.pt` (копия для приложения)

Можно также загрузить `.pt` файл через боковую панель в интерфейсе.

## Требования

- Python 3.10+
- GPU (рекомендуется для обучения), CPU подходит для inference
- ~2 GB свободного места для датасета и весов

## Ссылки

- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [NEU-DET Dataset (Kaggle)](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)
- [Reference Kaggle Notebook](https://www.kaggle.com/code/hareshkanaaramaraj/manufacturing-defect-detection-using-yolov8/notebook)
