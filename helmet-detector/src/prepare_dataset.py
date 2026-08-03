"""Converte as anotacoes Pascal VOC (XML) para o formato YOLO e organiza o
dataset em data/processed/images|labels/{train,val}.

Uso: python src/prepare_dataset.py
"""
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

CLASSES = ["helmet", "head", "person"]
VAL_SPLIT = 0.2
SEED = 42


def voc_object_to_yolo_line(obj, img_w, img_h):
    class_id = CLASSES.index(obj.find("name").text)
    box = obj.find("bndbox")
    xmin = float(box.find("xmin").text)
    ymin = float(box.find("ymin").text)
    xmax = float(box.find("xmax").text)
    ymax = float(box.find("ymax").text)

    x_center = ((xmin + xmax) / 2) / img_w
    y_center = ((ymin + ymax) / 2) / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def convert_annotation(xml_path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    img_w = float(size.find("width").text)
    img_h = float(size.find("height").text)
    return [voc_object_to_yolo_line(obj, img_w, img_h) for obj in root.findall("object")]


def main():
    annotations = sorted((RAW_DIR / "annotations").glob("*.xml"))
    random.Random(SEED).shuffle(annotations)

    split_idx = int(len(annotations) * (1 - VAL_SPLIT))
    splits = {"train": annotations[:split_idx], "val": annotations[split_idx:]}

    for split, files in splits.items():
        images_out = PROCESSED_DIR / "images" / split
        labels_out = PROCESSED_DIR / "labels" / split
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        for xml_path in files:
            stem = xml_path.stem
            image_path = RAW_DIR / "images" / f"{stem}.png"
            if not image_path.exists():
                print(f"aviso: imagem nao encontrada para {stem}, pulando")
                continue

            lines = convert_annotation(xml_path)
            (labels_out / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            shutil.copy2(image_path, images_out / image_path.name)

        print(f"{split}: {len(files)} imagens")


if __name__ == "__main__":
    main()
