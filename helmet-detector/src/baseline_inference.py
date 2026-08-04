"""Roda o YOLOv8n pre-treinado (COCO) em uma imagem de teste para confirmar
que a inferencia funciona no PC (CPU, sem GPU) e medir o tempo por imagem.

Uso: python src/baseline_inference.py
"""
import time
from pathlib import Path

from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "yolov8n.pt"
IMAGES_DIR = ROOT_DIR / "data" / "processed" / "images" / "train"
OUTPUT_DIR = ROOT_DIR / "outputs" / "baseline"

NUM_WARMUP = 1
NUM_RUNS = 10


def pick_test_image():
    images = sorted(IMAGES_DIR.glob("*.png"))
    if not images:
        raise SystemExit(f"nenhuma imagem encontrada em {IMAGES_DIR}")
    return images[0]


def main():
    image_path = pick_test_image()
    print(f"imagem de teste: {image_path}")

    model = YOLO(str(MODEL_PATH))

    # warm-up: a primeira inferencia inclui overhead de inicializacao (nao entra na media)
    for _ in range(NUM_WARMUP):
        model.predict(source=str(image_path), verbose=False)

    times = []
    for _ in range(NUM_RUNS):
        start = time.perf_counter()
        results = model.predict(source=str(image_path), verbose=False)
        times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times)
    print(f"tempo medio de inferencia ({NUM_RUNS} execucoes): {avg_time * 1000:.1f} ms/imagem")
    print(f"min: {min(times) * 1000:.1f} ms | max: {max(times) * 1000:.1f} ms")

    result = results[0]
    print(f"objetos detectados: {len(result.boxes)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"baseline_{image_path.stem}.png"
    result.save(filename=str(out_path))
    print(f"salvo: {out_path}")


if __name__ == "__main__":
    main()
