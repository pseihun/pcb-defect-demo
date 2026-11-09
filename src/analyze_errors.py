"""Error analysis for one class (default: `short`) on the held-out test boards.

What it does
  1. Per-class P / R / AP50 / AP50-95 on val and test boards.
  2. Label statistics (count, box size) per split.
  3. False-positive breakdown on test: background / localization / class confusion / duplicate,
     per board, at conf >= 0.25 and >= 0.5.
  4. Recurring-FP check: finds the spot where most background FPs land on one board and
     recomputes AP50 with that spot treated as an "ignore region" (COCO-crowd style:
     predictions and labels inside it are dropped).
  5. Crops of the top FPs / TPs for visual inspection.

Usage (from the project root):
    python src/analyze_errors.py                        # -> analysis/report.json, analysis/*.jpg
    python src/analyze_errors.py --cls short --out analysis
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

NAMES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]


def load_gt(data: Path, split: str) -> dict:
    """stem -> (image path, w, h, [(cls, x1, y1, x2, y2), ...]) in pixels."""
    gt = {}
    for lbl in sorted((data / "labels" / split).glob("*.txt")):
        img = data / "images" / split / f"{lbl.stem}.jpg"
        h, w = cv2.imread(str(img)).shape[:2]
        boxes = []
        for line in lbl.read_text().splitlines():
            if line.strip():
                c, cx, cy, bw, bh = map(float, line.split())
                boxes.append((int(c), (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
        gt[lbl.stem] = (img, w, h, boxes)
    return gt


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def ap50(preds, gts) -> float:
    """AP at IoU 0.5 with 101-point interpolation (COCO style), greedy matching by confidence.

    preds: [(stem, conf, box)], gts: {stem: [box, ...]} for a single class.
    """
    n_gt = sum(len(v) for v in gts.values())
    if n_gt == 0:
        return 0.0
    used = {s: [False] * len(v) for s, v in gts.items()}
    tp = []
    for stem, _, box in sorted(preds, key=lambda p: -p[1]):
        best, j = 0.0, -1
        for k, g in enumerate(gts.get(stem, [])):
            v = iou(box, g)
            if v > best and not used[stem][k]:
                best, j = v, k
        if best >= 0.5:
            used[stem][j] = True
            tp.append(1)
        else:
            tp.append(0)
    tp = np.array(tp, float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # renamed in NumPy 2
    return float(trapezoid(np.interp(x, mrec, mpre), x))


def crops(records, gt, cls, n, color, path):
    """Save a 6-column montage of crops around the given predictions."""
    tiles = []
    for conf, kind, stem, box in records[:n]:
        img = cv2.imread(str(gt[stem][0]))
        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        half = max(x2 - x1, y2 - y1, 60) * 2
        X1, Y1 = max(cx - half, 0), max(cy - half, 0)
        crop = img[Y1:cy + half, X1:cx + half].copy()
        cv2.rectangle(crop, (x1 - X1, y1 - Y1), (x2 - X1, y2 - Y1), color, 2)
        for gc, *gb in gt[stem][3]:
            if gc == cls:
                gx1, gy1, gx2, gy2 = map(int, gb)
                cv2.rectangle(crop, (gx1 - X1, gy1 - Y1), (gx2 - X1, gy2 - Y1), (255, 0, 0), 1)
        crop = cv2.resize(crop, (240, 240))
        cv2.putText(crop, f"{stem[:2]} {conf:.2f} {kind[:5]}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        tiles.append(crop)
    while len(tiles) % 6:
        tiles.append(np.zeros((240, 240, 3), np.uint8))
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--data", default="data/yolo")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--cls", default="short")
    ap.add_argument("--out", default="analysis")
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out)
    out.mkdir(exist_ok=True)
    cls = NAMES.index(args.cls)
    model = YOLO(args.weights)
    report = {"class": args.cls, "weights": args.weights, "imgsz": args.imgsz}

    print("[1/5] label statistics", flush=True)
    report["label_stats"] = {}
    for split in ("train", "val", "test"):
        per = defaultdict(list)
        for *_, boxes in load_gt(data, split).values():
            for c, x1, y1, x2, y2 in boxes:
                per[NAMES[c]].append((x2 - x1, y2 - y1))
        report["label_stats"][split] = {
            n: {"count": len(v), "median_w": round(float(np.median([a for a, _ in v])), 1),
                "median_h": round(float(np.median([b for _, b in v])), 1)} for n, v in per.items()}

    print("[2/5] per-class metrics (val, test)", flush=True)
    for split in ("val", "test"):
        # workers=0: Windows spawns dataloader workers that would re-run this script otherwise.
        # save_json: keep the validator's own predictions, so every number below matches the official
        # metric. (model.predict() preprocesses slightly differently, and for `short` that alone moved
        # AP50 from 0.646 to 0.865 because its false positives score close to its true positives.)
        m = model.val(data=str(data / "pcb.yaml"), split=split, imgsz=args.imgsz, plots=False, verbose=False,
                      workers=0, save_json=split == "test", project="runs", name=f"analyze_{split}", exist_ok=True)
        report[f"per_class_{split}"] = {
            NAMES[c]: {"P": round(float(m.box.p[i]), 3), "R": round(float(m.box.r[i]), 3),
                       "AP50": round(float(m.box.ap50[i]), 3), "AP50_95": round(float(m.box.ap[i]), 3)}
            for i, c in enumerate(m.box.ap_class_index)}
    val_preds = defaultdict(list)                       # stem -> [(cls, conf, xyxy)]
    for d in json.loads((Path(m.save_dir) / "predictions.json").read_text()):
        x, y, w, h = d["bbox"]
        val_preds[str(d["image_id"])].append((d["category_id"] - 1, d["score"], [x, y, x + w, y + h]))

    print("[3/5] false-positive breakdown on test", flush=True)
    gt = load_gt(data, "test")
    all_preds = defaultdict(list)                       # cls -> [(stem, conf, box)]
    fp, tp = defaultdict(list), defaultdict(list)       # cls -> [(conf, kind, stem, box)]
    for stem, (img, _, _, boxes) in gt.items():
        preds = sorted(val_preds[stem], key=lambda p: -p[1])
        matched = defaultdict(set)
        for c, conf, box in preds:
            all_preds[c].append((stem, conf, box))
            same = [(iou(box, gb), j) for j, (gc, *gb) in enumerate(boxes) if gc == c]
            best_same, best_j = max(same, default=(0.0, None))
            best_other = max((iou(box, gb) for gc, *gb in boxes if gc != c), default=0.0)
            # Matching is per class, greedy by confidence (same rule as ap50()).
            if best_same >= 0.5 and best_j not in matched[c]:
                matched[c].add(best_j)
                tp[c].append((conf, "tp", stem, box))
                continue
            kind = ("duplicate" if best_same >= 0.5 else "class_confusion" if best_other >= 0.5
                    else "localization" if best_same >= 0.1 else "background")
            fp[c].append((conf, kind, stem, box))

    report["fp_breakdown_test"] = {}
    for c, name in enumerate(NAMES):
        report["fp_breakdown_test"][name] = {
            f"conf>={thr}": {
                "TP": sum(t[0] >= thr for t in tp[c]),
                "FP": sum(f[0] >= thr for f in fp[c]),
                "FP_by_type": dict(Counter(f[1] for f in fp[c] if f[0] >= thr)),
                "FP_by_board": dict(Counter(f[2][:2] for f in fp[c] if f[0] >= thr)),
            } for thr in (0.25, 0.5)}

    print("[4/5] recurring-FP ignore-region check", flush=True)
    bg = [f for f in fp[cls] if f[1] == "background" and f[0] >= 0.25]
    board = Counter(f[2][:2] for f in bg).most_common(1)[0][0]
    centers = np.array([((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for _, _, s, b in bg if s[:2] == board])
    med = np.median(centers, axis=0)
    near = centers[np.linalg.norm(centers - med, axis=1) < 150]
    pad = 40
    region = [float(near[:, 0].min() - pad), float(near[:, 1].min() - pad),
              float(near[:, 0].max() + pad), float(near[:, 1].max() + pad)]

    def inside(stem, box):
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        return stem[:2] == board and region[0] <= cx <= region[2] and region[1] <= cy <= region[3]

    gts = {s: [tuple(b) for c, *b in v[3] if c == cls] for s, v in gt.items()}
    gts_kept = {s: [b for b in v if not inside(s, b)] for s, v in gts.items()}
    preds_kept = [p for p in all_preds[cls] if not inside(p[0], p[2])]
    report["recurring_fp"] = {
        "board": board,
        "region_xyxy_px": [round(v) for v in region],
        "bg_fp_on_board_conf>=0.25": int(len(centers)),
        "bg_fp_in_region_conf>=0.25": int(len(near)),
        "center_spread_px_std": [round(float(v), 1) for v in near.std(axis=0)],
        "gt_in_region": sum(len(gts[s]) - len(gts_kept[s]) for s in gts),
        "preds_removed": len(all_preds[cls]) - len(preds_kept),
        "AP50_all (own calc)": round(ap50(all_preds[cls], gts), 3),
        "AP50_ignore_region (own calc)": round(ap50(preds_kept, gts_kept), 3),
    }

    print("[5/5] crops", flush=True)
    crops(sorted(fp[cls], key=lambda f: -f[0]), gt, cls, 24, (0, 0, 255), out / f"{args.cls}_top_fp.jpg")
    crops(sorted(tp[cls], key=lambda t: -t[0]), gt, cls, 12, (0, 200, 0), out / f"{args.cls}_top_tp.jpg")
    report["tp_iou_median"] = {
        NAMES[c]: round(float(np.median([max(iou(t[3], gb) for gc, *gb in gt[t[2]][3] if gc == c) for t in tp[c]])), 3)
        for c in range(len(NAMES)) if tp[c]}

    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report["recurring_fp"], indent=2))
    print(f"wrote {out / 'report.json'}")


if __name__ == "__main__":  # required on Windows
    main()
