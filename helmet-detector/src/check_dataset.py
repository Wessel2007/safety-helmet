"""Abre algumas imagens do dataset processado com as bounding boxes desenhadas,
so para conferir visualmente que as anotacoes (convertidas para YOLO) estao corretas.

Uso: python src/check_dataset.py
"""
import random
from pathlib import Path

import cv2
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT_DIR / "data.yaml"
PREVIEW_DIR = ROOT_DIR / "outputs" / "dataset_preview"

SPLIT = "train"
NUM_SAMPLES = 3
SEED = 42

BOX_COLORS = {
    "helmet": (0, 200, 0),
    "head": (0, 0, 255),
    "person": (255, 150, 0),
}


def load_classes():
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["names"]


def draw_boxes(image_path, label_path, classes):
    image = cv2.imread(str(image_path))
    img_h, img_w = image.shape[:2]

    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            class_id, xc, yc, bw, bh = line.split()
            name = classes[int(class_id)]

            xc, yc = float(xc) * img_w, float(yc) * img_h
            bw, bh = float(bw) * img_w, float(bh) * img_h
            xmin, ymin = int(xc - bw / 2), int(yc - bh / 2)
            xmax, ymax = int(xc + bw / 2), int(yc + bh / 2)

            color = BOX_COLORS.get(name, (255, 255, 255))
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(image, name, (xmin, max(ymin - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return image


def main():
    classes = load_classes()
    images_dir = ROOT_DIR / "data" / "processed" / "images" / SPLIT
    labels_dir = ROOT_DIR / "data" / "processed" / "labels" / SPLIT

    images = sorted(images_dir.glob("*.png"))
    if not images:
        raise SystemExit(
            f"nenhuma imagem encontrada em {images_dir} — rode src/prepare_dataset.py antes"
        )

    samples = random.Random(SEED).sample(images, min(NUM_SAMPLES, len(images)))

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for image_path in samples:
        label_path = labels_dir / f"{image_path.stem}.txt"
        annotated = draw_boxes(image_path, label_path, classes)
        out_path = PREVIEW_DIR / f"preview_{image_path.stem}.png"
        cv2.imwrite(str(out_path), annotated)
        print(f"salvo: {out_path}")


if __name__ == "__main__":
    main()
