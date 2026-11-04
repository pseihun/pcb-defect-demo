# PCB Defect Detection Demo

> 🚧 Work in progress

Upload a PCB image and get detected defects (missing hole, short, open circuit, spur, etc.) with bounding boxes.

## Results (YOLO11s, imgsz 1280, test on unseen boards 10 & 12)

| Class | mAP50 |
|---|---|
| missing_hole | 0.995 |
| mouse_bite | 0.985 |
| open_circuit | 0.994 |
| short | **0.646** |
| spur | 0.983 |
| spurious_copper | 0.977 |
| **all** | **0.930** (mAP50-95 0.481) |

Inference ≈ 12 ms / image on an RTX 4090. The `short` class is the weak spot (precision 0.52), which suggests many
false positives on boards not seen in training.

## Reproduce
```bash
python -m venv .venv && .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# prepare data: see data/README.md
python src/train.py          # trains, then evaluates on test boards -> results/pcb_yolo11s.json
```

## Goals
- [x] Prepare a public PCB defect dataset (board-level split, no leakage)
- [x] Train a YOLO-based detector
- [x] Evaluate (mAP, per-class results)
- [ ] Error analysis (why is `short` weak?)
- [x] Build a Gradio web demo (`python app.py`)
- [ ] Deploy to Hugging Face Spaces

## Tech Stack
Python · PyTorch · Ultralytics YOLO · Gradio

## Project Structure
```
pcb-defect-demo/
├── data/        # dataset (not tracked by git, see data/README.md)
├── notebooks/   # exploration & experiments
└── src/         # training / inference / app code
```
