"""Evaluate a trained checkpoint on the held-out test boards and save metrics as JSON.

Usage:
    python src/evaluate.py --weights runs/detect/runs/pcb_yolo11s/weights/best.pt
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def evaluate(weights: str | Path, data: str, imgsz: int, name: str) -> dict:
    metrics = YOLO(weights).val(data=data, split="test", imgsz=imgsz, project="runs", name=f"{name}_test", exist_ok=True)
    names = metrics.names
    result = {
        "weights": Path(weights).as_posix(),
        "imgsz": imgsz,
        "test_mAP50": round(float(metrics.box.map50), 4),
        "test_mAP50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "per_class_mAP50": {names[c]: round(float(metrics.box.ap50[i]), 4) for i, c in enumerate(metrics.box.ap_class_index)},
    }
    out = Path("results") / f"{name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="data/yolo/pcb.yaml")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--name", default="pcb_yolo11s")
    args = ap.parse_args()
    evaluate(args.weights, args.data, args.imgsz, args.name)


if __name__ == "__main__":
    main()
