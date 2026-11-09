"""Gradio web demo: upload a PCB image, get detected defects.

Run locally:
    python app.py            # then open http://127.0.0.1:7860
"""

import os
from pathlib import Path

import gradio as gr
from ultralytics import YOLO

MODEL_PATH = os.getenv("MODEL_PATH", "weights/best.pt")
IMGSZ = 1280
model = YOLO(MODEL_PATH)


def detect(image, conf: float):
    if image is None:
        return None, []
    # Gradio gives RGB, but Ultralytics treats numpy arrays as BGR (OpenCV order).
    result = model.predict(image[:, :, ::-1], imgsz=IMGSZ, conf=conf, verbose=False)[0]
    annotated = result.plot()[:, :, ::-1]  # BGR -> RGB for display

    rows = []
    for box in result.boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        rows.append([result.names[int(box.cls)], round(float(box.conf), 3), x1, y1, x2, y2])
    rows.sort(key=lambda r: -r[1])
    return annotated, rows


examples = [[str(p), 0.25] for p in sorted(Path("examples").glob("*.jpg"))]

with gr.Blocks(title="PCB Defect Detection") as demo:
    gr.Markdown(
        "# PCB Defect Detection\n"
        "Upload a bare-PCB image to detect 6 defect types: "
        "missing hole, mouse bite, open circuit, short, spur, spurious copper.  \n"
        "Model: YOLO11s trained on the PKU PCB Defects dataset (test mAP50 0.93 on unseen boards)."
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="numpy", label="PCB image")
            conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence threshold")
            btn = gr.Button("Detect", variant="primary")
        with gr.Column():
            out_img = gr.Image(label="Detections")
            out_tbl = gr.Dataframe(headers=["defect", "confidence", "x1", "y1", "x2", "y2"], label="Detected defects")
    gr.Examples(examples=examples, inputs=[inp, conf])
    btn.click(detect, inputs=[inp, conf], outputs=[out_img, out_tbl])

if __name__ == "__main__":
    demo.launch()
