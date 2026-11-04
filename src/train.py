"""Train a YOLO detector on the PCB defect dataset, then evaluate on the held-out test boards.

Usage:
    python src/train.py                       # defaults below
    python src/train.py --model yolo11m.pt --epochs 150
"""

import argparse

from ultralytics import YOLO

from evaluate import evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/yolo/pcb.yaml")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=100)
    # Defects are tiny (~2% of image width) on ~3000px boards, so use a large input size.
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", default="pcb_yolo11s")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=30,
        seed=args.seed,
        deterministic=True,
        project="runs",
        name=args.name,
        exist_ok=True,
        flipud=0.5,  # PCB images have no canonical "up"
    )

    # Evaluate the best checkpoint on unseen test boards.
    # (Ultralytics decides the final save dir, so ask the trainer instead of guessing the path.)
    evaluate(model.trainer.best, args.data, args.imgsz, args.name)


if __name__ == "__main__":  # required on Windows for dataloader workers
    main()
