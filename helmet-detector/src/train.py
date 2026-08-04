"""Fine-tuning do YOLOv8n no dataset de capacetes (helmet/head/person).

Hiperparametros pensados para treino em CPU (sem GPU dedicada): imagem menor,
batch pequeno e early stopping via `patience` para nao gastar horas a toa.

Uso:
    python src/train.py
    python src/train.py --epochs 30 --imgsz 512 --batch 8
    python src/train.py --resume
"""
import argparse
import os
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT_DIR / "data.yaml"
BASE_MODEL = ROOT_DIR / "models" / "yolov8n.pt"
RUNS_DIR = ROOT_DIR / "outputs" / "train"
MODELS_DIR = ROOT_DIR / "models"
RUN_NAME = "helmet_yolov8n"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15, help="epocas sem melhora antes de parar")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true", help="retoma o ultimo treino interrompido")
    return parser.parse_args()


def main():
    args = parse_args()

    # o campo "path" do data.yaml e relativo ao diretorio de execucao (nao ao
    # arquivo yaml), entao garantimos que o treino sempre rode a partir da
    # raiz do projeto para resolver data/processed corretamente.
    os.chdir(ROOT_DIR)

    last_weights = RUNS_DIR / RUN_NAME / "weights" / "last.pt"
    if args.resume and last_weights.exists():
        print(f"retomando treino a partir de: {last_weights}")
        model = YOLO(str(last_weights))
        model.train(resume=True)
    else:
        model = YOLO(str(BASE_MODEL))
        model.train(
            data=str(DATA_YAML),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            workers=args.workers,
            device=args.device,
            project=str(RUNS_DIR),
            name=RUN_NAME,
            exist_ok=True,
        )

    best_weights = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
    if best_weights.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        dest = MODELS_DIR / "helmet_yolov8n_best.pt"
        shutil.copy2(best_weights, dest)
        print(f"modelo treinado salvo em: {dest}")
    else:
        print(f"aviso: pesos nao encontrados em {best_weights}")


if __name__ == "__main__":
    main()
