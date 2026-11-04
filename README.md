# PCB Defect Detection Demo

> 🚧 Work in progress

Upload a PCB image and get detected defects (missing hole, short, open circuit, spur, etc.) with bounding boxes.

## Goals
- [ ] Prepare a public PCB defect dataset
- [ ] Train a YOLO-based detector
- [ ] Evaluate (mAP, per-class results)
- [ ] Build a Gradio web demo
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
