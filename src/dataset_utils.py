from __future__ import annotations

import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

from src.constants import DEFECT_CLASSES, PROJECT_ROOT


def xml_to_yolo_line(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    img_w: int,
    img_h: int,
    class_id: int,
) -> str:
    x_center = ((xmin + xmax) / 2) / img_w
    y_center = ((ymin + ymax) / 2) / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _build_image_index(images_dir: Path) -> dict[str, Path]:
    """Индекс всех изображений, включая вложенные папки по классам."""
    index: dict[str, Path] = {}
    for path in images_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        index.setdefault(path.name.lower(), path)
        index.setdefault(path.stem.lower(), path)
    return index


def _find_image_file(
    images_dir: Path,
    filename: str,
    xml_stem: str,
    image_index: dict[str, Path] | None = None,
) -> Path | None:
    """Ищет изображение в images/ и во вложенных папках классов."""
    index = image_index or _build_image_index(images_dir)

    for key in (filename.lower(), xml_stem.lower()):
        if key in index:
            return index[key]

    # scratches_300 -> images/scratches/scratches_300.jpg
    if "_" in xml_stem:
        class_part = xml_stem.rsplit("_", 1)[0]
        for ext in IMAGE_EXTENSIONS:
            nested = images_dir / class_part / f"{xml_stem}{ext}"
            if nested.exists():
                return nested
            nested = images_dir / class_part / filename
            if nested.exists():
                return nested

    return None


def _find_split_dirs(raw_dir: Path) -> dict[str, Path] | None:
    """Ищет уже разбитый датасет: NEU-DET/train|validation/{images,annotations}."""
    split_aliases = {
        "train": ("train",),
        "val": ("val", "validation", "valid", "valdation"),
        "test": ("test",),
    }

    found: dict[str, Path] = {}
    for split_key, names in split_aliases.items():
        for name in names:
            candidate = raw_dir / name
            if not candidate.is_dir():
                continue
            images = candidate / "images"
            annotations = candidate / "annotations"
            if images.is_dir() and annotations.is_dir():
                found[split_key] = candidate
                break
            # альтернатива: IMAGES / ANNOTATIONS
            images = candidate / "IMAGES"
            annotations = candidate / "ANNOTATIONS"
            if images.is_dir() and annotations.is_dir():
                found[split_key] = candidate
                break

    if "train" in found and ("val" in found or "test" in found):
        return found
    if "train" in found and len(found) == 1:
        return found
    return None


def _resolve_image_label_dirs(split_dir: Path) -> tuple[Path, Path]:
    for images_name, labels_name in (
        ("images", "annotations"),
        ("IMAGES", "ANNOTATIONS"),
    ):
        images_dir = split_dir / images_name
        annotations_dir = split_dir / labels_name
        if images_dir.is_dir() and annotations_dir.is_dir():
            return images_dir, annotations_dir
    raise FileNotFoundError(
        f"В {split_dir} не найдены пары images+annotations или IMAGES+ANNOTATIONS"
    )


def convert_neu_det_xml_to_yolo(
    source_dir: Path,
    output_dir: Path,
    class_names: list[str] | None = None,
    images_dir: Path | None = None,
    annotations_dir: Path | None = None,
) -> dict[str, int]:
    """Конвертирует XML-аннотации NEU-DET в формат YOLO."""
    class_names = class_names or DEFECT_CLASSES
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    if images_dir is None or annotations_dir is None:
        images_dir = source_dir / "IMAGES"
        annotations_dir = source_dir / "ANNOTATIONS"
        if not images_dir.exists():
            images_dir = source_dir / "images"
        if not annotations_dir.exists():
            annotations_dir = source_dir / "annotations"

    output_images = output_dir / "images"
    output_labels = output_dir / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    image_index = _build_image_index(images_dir)
    converted = 0
    for xml_path in sorted(annotations_dir.glob("*.xml")):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.findtext("filename") or f"{xml_path.stem}.jpg"
        size = root.find("size")
        if size is None:
            continue

        img_w = int(size.findtext("width", "200"))
        img_h = int(size.findtext("height", "200"))

        lines: list[str] = []
        for obj in root.findall("object"):
            name = obj.findtext("name")
            if name not in class_to_id:
                continue

            bbox = obj.find("bndbox")
            if bbox is None:
                continue

            xmin = float(bbox.findtext("xmin", "0"))
            ymin = float(bbox.findtext("ymin", "0"))
            xmax = float(bbox.findtext("xmax", "0"))
            ymax = float(bbox.findtext("ymax", "0"))

            lines.append(
                xml_to_yolo_line(xmin, ymin, xmax, ymax, img_w, img_h, class_to_id[name])
            )

        if not lines:
            continue

        image_src = _find_image_file(images_dir, filename, xml_path.stem, image_index)
        if image_src is None:
            continue

        shutil.copy2(image_src, output_images / image_src.name)
        (output_labels / f"{xml_path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        converted += 1

    return {"converted": converted, "classes": len(class_names)}


def convert_presplit_neu_det_to_yolo(
    raw_dir: Path,
    output_root: Path,
    class_names: list[str] | None = None,
) -> dict[str, int]:
    """Конвертирует NEU-DET с готовыми train/validation/test в YOLO-формат."""
    splits = _find_split_dirs(raw_dir)
    if not splits:
        raise FileNotFoundError(
            "Не найдена структура train/validation с папками images и annotations"
        )

    dataset_root = output_root / "split"
    counts: dict[str, int] = {}

    for split_key, split_dir in splits.items():
        images_dir, annotations_dir = _resolve_image_label_dirs(split_dir)
        result = convert_neu_det_xml_to_yolo(
            source_dir=split_dir,
            output_dir=dataset_root / split_key,
            class_names=class_names,
            images_dir=images_dir,
            annotations_dir=annotations_dir,
        )
        counts[split_key] = result["converted"]

    return counts


def split_yolo_dataset(
    source_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, int]:
    """Разбивает YOLO-датасет на train/val/test."""
    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels"

    pairs = []
    for image_path in sorted(images_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append((image_path, label_path))

    random.seed(seed)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train : n_train + n_val],
        "test": pairs[n_train + n_val :],
    }

    counts: dict[str, int] = {}
    for split_name, items in splits.items():
        split_images = output_dir / split_name / "images"
        split_labels = output_dir / split_name / "labels"
        split_images.mkdir(parents=True, exist_ok=True)
        split_labels.mkdir(parents=True, exist_ok=True)

        for image_path, label_path in items:
            shutil.copy2(image_path, split_images / image_path.name)
            shutil.copy2(label_path, split_labels / f"{image_path.stem}.txt")

        counts[split_name] = len(items)

    return counts


def write_data_yaml(
    dataset_root: Path,
    output_path: Path,
    class_names: list[str] | None = None,
    include_test: bool = True,
) -> Path:
    class_names = class_names or DEFECT_CLASSES
    data = {
        "path": str(dataset_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": len(class_names),
        "names": class_names,
    }
    if include_test and (dataset_root / "test" / "images").exists():
        data["test"] = "test/images"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return output_path


def prepare_neu_det_pipeline(
    raw_dir: Path,
    output_root: Path | None = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Path:
    """Полный пайплайн подготовки NEU-DET для YOLOv8."""
    output_root = output_root or (PROJECT_ROOT / "data" / "neu_det_yolo")
    dataset_root = output_root / "split"

    presplit = _find_split_dirs(raw_dir)
    if presplit:
        counts = convert_presplit_neu_det_to_yolo(raw_dir, output_root)
        has_test = (dataset_root / "test" / "images").exists()
    else:
        yolo_raw = output_root / "all"
        convert_neu_det_xml_to_yolo(raw_dir, yolo_raw)
        counts = split_yolo_dataset(yolo_raw, dataset_root, train_ratio, val_ratio)
        has_test = True

    total = sum(counts.values()) if isinstance(counts, dict) else 0
    if total == 0:
        raise FileNotFoundError(
            "Не удалось подготовить датасет: 0 изображений сконвертировано. "
            "Проверьте, что в images/ (или images/<класс>/) лежат .jpg/.png файлы, "
            "соответствующие XML в annotations/."
        )

    yaml_path = output_root / "data.yaml"
    write_data_yaml(dataset_root, yaml_path, include_test=has_test)

    config_path = PROJECT_ROOT / "config" / "neu_det.yaml"
    write_data_yaml(dataset_root, config_path, include_test=has_test)

    return yaml_path
