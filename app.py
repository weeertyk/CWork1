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
    DEFAULT_MODEL,
    YOLO_BASE_MODELS,
)
from src.detector import DefectDetector  # noqa: E402
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
            "Используется базовая COCO-модель. Загрузите обученные веса (.pt) "
            "или поместите best.pt в папку models/."
        )

    st.sidebar.divider()
    st.sidebar.markdown("**Классы дефектов (NEU-DET):**")
    for cls in DEFECT_CLASSES:
        st.sidebar.text(f"• {DEFECT_CLASSES_RU.get(cls, cls)}")

    return selected_model, conf, iou, device_arg


@st.cache_resource(show_spinner="Загрузка модели...")
def load_detector(model_path: str, conf: float, iou: float, device: str | None) -> DefectDetector:
    return DefectDetector(model_path=model_path, conf=conf, iou=iou, device=device)


def run_analysis(
    sources: list[tuple[Image.Image, str]],
    model_path: str,
    conf: float,
    iou: float,
    device: str | None,
) -> None:
    detector = load_detector(model_path, conf, iou, device)
    results = detector.predict_batch(
        [img for img, _ in sources],
        [name for _, name in sources],
    )
    st.session_state.analysis_results = results


def render_results() -> None:
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


def tab_analysis(model_path: str, conf: float, iou: float, device: str | None) -> None:
    st.subheader("Анализ изображений")

    tab_upload, tab_camera = st.tabs(["📁 Загрузка файлов", "📸 Камера"])

    with tab_upload:
        st.markdown("Загрузите одно или несколько изображений металлической детали.")
        uploaded = st.file_uploader(
            "Выберите файлы",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            accept_multiple_files=True,
            key="upload_files",
        )
        st.info("Поддерживается пакетная загрузка. Модель выделяет дефекты и формирует отчёт.")

        if st.button("🔍 Запустить анализ", type="primary", key="analyze_upload"):
            if not uploaded:
                st.warning("Загрузите хотя бы одно изображение.")
            else:
                sources = [(Image.open(file), file.name) for file in uploaded]
                run_analysis(sources, model_path, conf, iou, device)

    with tab_camera:
        st.markdown("Сделайте снимок с камеры устройства.")
        camera = st.camera_input("Камера", key="camera_input")
        st.info("Снимок будет проанализирован на наличие дефектов поверхности.")

        if st.button("🔍 Запустить анализ", type="primary", key="analyze_camera"):
            if not camera:
                st.warning("Сделайте снимок с камеры.")
            else:
                sources = [(Image.open(camera), "camera.jpg")]
                run_analysis(sources, model_path, conf, iou, device)

    render_results()


def main() -> None:
    init_session_state()

    st.title("🔍 Анализ дефектов металлических деталей")
    st.caption("Computer Vision · YOLOv8 · NEU-DET · Streamlit")

    model_path, conf, iou, device = sidebar_settings()
    tab_analysis(model_path, conf, iou, device)


if __name__ == "__main__":
    main()
