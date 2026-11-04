"""Convert the PKU PCB Defects dataset (Pascal VOC XML) to YOLO format.

Split is done by *base board ID* (the `NN_` filename prefix), not by image,
so the same physical board never appears in both train and test. A random
image-level split would leak board appearance across splits and inflate mAP.

Usage:
    python src/prepare_data.py --src data/raw/PCB_DATASET --dst data/yolo
"""

import argparse
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

CLASSES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]

# Board IDs present: 01, 04, 05, 06, 07, 08, 09, 10, 11, 12
SPLITS = {
    "train": {"01", "04", "05", "06", "07", "08"},
    "val": {"09", "11"},
    "test": {"10", "12"},
}


def board_id(stem: str) -> str:
    return stem.split("_")[0]


def split_of(stem: str) -> str:
    bid = board_id(stem)
    for name, boards in SPLITS.items():
        if bid in boards:
            return name
    raise ValueError(f"Board {bid} not assigned to any split")


def voc_to_yolo(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    w = float(root.findtext("size/width"))
    h = float(root.findtext("size/height"))
    lines = []
    for obj in root.iter("object"):
        cls = CLASSES.index(obj.findtext("name").strip().lower())
        b = obj.find("bndbox")
        x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/raw/PCB_DATASET"))
    ap.add_argument("--dst", type=Path, default=Path("data/yolo"))
    args = ap.parse_args()

    if args.dst.exists():
        shutil.rmtree(args.dst)

    img_count, obj_count = Counter(), Counter()
    for xml_path in sorted((args.src / "Annotations").rglob("*.xml")):
        img_path = args.src / "images" / xml_path.parent.name / f"{xml_path.stem}.jpg"
        if not img_path.exists():
            print(f"skip (no image): {img_path}")
            continue

        split = split_of(xml_path.stem)
        (args.dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.dst / "labels" / split).mkdir(parents=True, exist_ok=True)

        shutil.copy2(img_path, args.dst / "images" / split / img_path.name)
        lines = voc_to_yolo(xml_path)
        (args.dst / "labels" / split / f"{xml_path.stem}.txt").write_text("\n".join(lines) + "\n")

        img_count[split] += 1
        obj_count[split] += len(lines)

    yaml = [f"path: {args.dst.resolve().as_posix()}"]
    yaml += [f"{s}: images/{s}" for s in SPLITS]
    yaml += ["names:"] + [f"  {i}: {c}" for i, c in enumerate(CLASSES)]
    (args.dst / "pcb.yaml").write_text("\n".join(yaml) + "\n")

    for s in SPLITS:
        print(f"{s:5s}: {img_count[s]:4d} images, {obj_count[s]:5d} objects, boards={sorted(SPLITS[s])}")
    print(f"wrote {args.dst / 'pcb.yaml'}")


if __name__ == "__main__":
    main()
