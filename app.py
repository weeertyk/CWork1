"""Streamlit-приложение для анализа дефектов на металлических деталях."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import (  # noqa: E402
    DEFECT_CLASSES,
    DEFECT_CLASSES_RU,
    DEFAULT_DATA_YAML,
    DEFAULT_MODEL,
    YOLO_BASE_MODELS,
)
from src.dataset_utils import prepare_neu_det_pipeline, write_data_yaml  # noqa: E402
from src.detector import DefectDetector  # noqa: E402
from src.trainer import DefectTrainer, TrainingConfig  # noqa: E402
from src.visualization import (  # noqa: E402
    bgr_to_rgb,
    detections_to_dataframe,
    plot_class_distribution,
    results_summary_dataframe,
)

st.set_page_config(
    page_title="Metal Defect Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .verdict-ok { color: #27ae60; font-size: 1.4rem; font-weight: 700; }
    .verdict-defect { color: #e74c3c; font-size: 1.4rem; font-weight: 700; }
    .block-container { padding-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state() -> None:
    defaults = {
        "analysis_results": [],
        "last_model_path": None,
        "training_done": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def sidebar_settings() -> tuple[str | None, float, float, str | None]:
    st.sidebar.header("⚙️ Настройки модели")

    model_options: list[str] = []
    if DEFAULT_MODEL.exists():
        model_options.append(str(DEFAULT_MODEL))
    latest = DefectDetector.find_latest_trained_weights()
    if latest and str(latest) not in model_options:
        model_options.append(str(latest))
    for base in YOLO_BASE_MODELS:
        if base not in model_options:
            model_options.append(base)

    selected_model = st.sidebar.selectbox("Модель", model_options, index=0)
    conf = st.sidebar.slider("Порог уверенности", 0.05, 0.95, 0.25, 0.05)
    iou = st.sidebar.slider("IoU (NMS)", 0.1, 0.9, 0.45, 0.05)
    device = st.sidebar.selectbox("Устройство", ["auto", "cpu", "0"], index=0)
    device_arg = None if device == "auto" else device

    custom_weights = st.sidebar.file_uploader(
        "Загрузить свои веса (.pt)", type=["pt"]
    )
    if custom_weights:
        weights_dir = ROOT / "models" / "uploaded"
        weights_dir.mkdir(parents=True, exist_ok=True)
        custom_path = weights_dir / custom_weights.name
        custom_path.write_bytes(custom_weights.getbuffer())
        selected_model = str(custom_path)

    if selected_model in YOLO_BASE_MODELS and not DEFAULT_MODEL.exists() and not latest:
        st.sidebar.warning(
            "Используется базовая COCO-модель. Для дефектов металла "
            "сначала обучите модель на вкладке «Обучение»."
        )

    st.sidebar.divider()
    st.sidebar.markdown("**Классы дефектов (NEU-DET):**")
    for cls in DEFECT_CLASSES:
        st.sidebar.text(f"• {DEFECT_CLASSES_RU.get(cls, cls)}")

    return selected_model, conf, iou, device_arg


@st.cache_resource(show_spinner="Загрузка модели...")
def load_detector(model_path: str, conf: float, iou: float, device: str | None) -> DefectDetector:
    return DefectDetector(model_path=model_path, conf=conf, iou=iou, device=device)


def tab_analysis(model_path: str, conf: float, iou: float, device: str | None) -> None:
    st.subheader("Анализ изображений")

    col_upload, col_info = st.columns([2, 1])
    with col_upload:
        uploaded = st.file_uploader(
            "Загрузите фото металлической детали",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            accept_multiple_files=True,
        )
    with col_info:
        st.info(
            "Поддерживается пакетная загрузка. "
            "Модель выделяет дефекты bounding box'ами и формирует отчёт."
        )

    camera = st.camera_input("Или сделайте снимок с камеры")
    sources: list[tuple[Image.Image, str]] = []

    if uploaded:
        for file in uploaded:
            sources.append((Image.open(file), file.name))
    if camera:
        sources.append((Image.open(camera), "camera.jpg"))

    if not sources:
        st.markdown(
            """
            ### Как начать
            1. Скачайте [NEU Surface Defect Database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) с Kaggle
            2. Подготовьте датасет на вкладке **Подготовка данных**
            3. Обучите модель на вкладке **Обучение**
            4. Загрузите изображение для анализа
            """
        )
        return

    if st.button("🔍 Запустить анализ", type="primary", use_container_width=False):
        detector = load_detector(model_path, conf, iou, device)
        results = detector.predict_batch(
            [img for img, _ in sources],
            [name for _, name in sources],
        )
        st.session_state.analysis_results = results

    results = st.session_state.analysis_results
    if not results:
        return

    summary_df = results_summary_dataframe(results)
    c1, c2, c3 = st.columns(3)
    c1.metric("Изображений", len(results))
    c2.metric("С дефектами", sum(1 for r in results if r.has_defects))
    c3.metric("Всего дефектов", sum(r.defect_count for r in results))

    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    chart = plot_class_distribution(results)
    if chart:
        st.image(chart, caption="Распределение типов дефектов")

    st.divider()
    for result in results:
        with st.expander(f"{result.source_name} — {'Брак' if result.has_defects else 'OK'}", expanded=True):
            col_orig, col_ann = st.columns(2)
            with col_orig:
                st.image(bgr_to_rgb(result.image_bgr), caption="Оригинал", use_container_width=True)
            with col_ann:
                st.image(bgr_to_rgb(result.annotated_bgr), caption="Детекция", use_container_width=True)

            if result.has_defects:
                st.markdown(
                    f'<p class="verdict-defect">⚠ Брак: обнаружено {result.defect_count} дефект(ов)</p>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<p class="verdict-ok">✓ Дефекты не обнаружены</p>', unsafe_allow_html=True)

            st.dataframe(
                detections_to_dataframe(result.detections),
                use_container_width=True,
                hide_index=True,
            )

    csv = summary_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 Скачать отчёт (CSV)",
        csv,
        file_name="defect_report.csv",
        mime="text/csv",
    )


def tab_training() -> None:
    st.subheader("Обучение YOLOv8")

    data_yaml = st.text_input("Путь к data.yaml", value=str(DEFAULT_DATA_YAML))
    base_model = st.selectbox("Базовая модель", YOLO_BASE_MODELS, index=0)
    col1, col2, col3 = st.columns(3)
    epochs = col1.number_input("Эпохи", 1, 500, 50)
    imgsz = col2.selectbox("Размер изображения", [320, 416, 640, 1280], index=2)
    batch = col3.number_input("Batch size", 1, 64, 16)
    device = st.selectbox("Устройство (обучение)", ["auto", "cpu", "0"], index=0)
    device_arg = None if device == "auto" else device
    run_name = st.text_input("Имя эксперимента", value="metal_defect_train")

    st.markdown(
        """
        Обучение основано на подходе из
        [Kaggle notebook (YOLOv8 manufacturing defects)](https://www.kaggle.com/code/hareshkanaaramaraj/manufacturing-defect-detection-using-yolov8/notebook).
        Для металлических деталей используется датасет **NEU-DET** (6 классов дефектов).
        """
    )

    if st.button("🚀 Начать обучение", type="primary"):
        yaml_path = Path(data_yaml)
        if not yaml_path.exists():
            st.error(f"Файл не найден: {yaml_path}. Сначала подготовьте датасет.")
            return

        progress = st.progress(0, text="Инициализация...")
        status = st.empty()

        try:
            trainer = DefectTrainer()
            status.info("Обучение запущено. Это может занять от нескольких минут до часов.")
            progress.progress(10, text="Обучение YOLOv8...")

            result = trainer.train(
                TrainingConfig(
                    data_yaml=yaml_path,
                    base_model=base_model,
                    epochs=int(epochs),
                    imgsz=int(imgsz),
                    batch=int(batch),
                    device=device_arg,
                    name=run_name,
                )
            )
            progress.progress(100, text="Готово!")
            st.session_state.training_done = True
            st.success(f"Обучение завершено. Веса: `{result.best_weights}`")
            st.info(f"Модель скопирована в `{DEFAULT_MODEL}` для использования в анализе.")

            val_metrics = trainer.validate(result.best_weights, yaml_path, device_arg)
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("mAP@0.5", f"{val_metrics['mAP50']:.3f}")
            mc2.metric("mAP@0.5:0.95", f"{val_metrics['mAP50-95']:.3f}")
            mc3.metric("Precision", f"{val_metrics['precision']:.3f}")
            mc4.metric("Recall", f"{val_metrics['recall']:.3f}")

            load_detector.clear()
        except Exception as exc:
            progress.empty()
            st.error(f"Ошибка обучения: {exc}")


def tab_dataset() -> None:
    st.subheader("Подготовка данных NEU-DET")

    st.markdown(
        """
        1. Скачайте датасет с Kaggle: [NEU Surface Defect Database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)
        2. Распакуйте в `data/NEU-DET/` — поддерживаются обе структуры:

        **Вариант A (ваш):**
        ```
        NEU-DET/
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

        **Вариант B (классический):**
        ```
        NEU-DET/
        ├── IMAGES/
        └── ANNOTATIONS/
        ```
        3. Укажите путь к папке `NEU-DET` ниже
        """
    )

    raw_dir = st.text_input(
        "Путь к исходному NEU-DET",
        value=str(ROOT / "data" / "NEU-DET"),
        placeholder=r"F:\chrevoygodie\CWork1\data\NEU-DET",
    )
    output_dir = st.text_input(
        "Папка для YOLO-датасета",
        value=str(ROOT / "data" / "neu_det_yolo"),
    )
    c1, c2 = st.columns(2)
    train_ratio = c1.slider("Train", 0.5, 0.9, 0.8, 0.05)
    val_ratio = c2.slider("Val", 0.05, 0.3, 0.1, 0.05)

    if st.button("📦 Подготовить датасет", type="primary"):
        raw_path = Path(raw_dir)
        if not raw_path.exists():
            st.error("Указанная папка не существует.")
            return

        with st.spinner("Конвертация XML → YOLO..."):
            yaml_path = prepare_neu_det_pipeline(
                raw_dir=raw_path,
                output_root=Path(output_dir),
                train_ratio=train_ratio,
                val_ratio=val_ratio,
            )
            config_path = ROOT / "config" / "neu_det.yaml"
            write_data_yaml(
                dataset_root=Path(output_dir) / "split",
                output_path=config_path,
                include_test=(Path(output_dir) / "split" / "test" / "images").exists(),
            )

        st.success("Датасет подготовлен!")
        st.code(str(yaml_path.resolve()))
        st.info(f"Конфиг обновлён: `{config_path}`")


def main() -> None:
    init_session_state()

    st.title("🔍 Анализ дефектов металлических деталей")
    st.caption("Computer Vision · YOLOv8 · NEU-DET · Streamlit")

    model_path, conf, iou, device = sidebar_settings()

    tab1, tab2, tab3 = st.tabs(["📷 Анализ", "🎓 Обучение", "📁 Подготовка данных"])

    with tab1:
        tab_analysis(model_path, conf, iou, device)
    with tab2:
        tab_training()
    with tab3:
        tab_dataset()


if __name__ == "__main__":
    main()
