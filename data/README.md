# Data

Dataset files are not committed to git. To reproduce:

```bash
# 1. Download (requires a Kaggle API token in ~/.kaggle/)
kaggle datasets download akhatova/pcb-defects -p data/raw --unzip

# 2. Convert VOC XML -> YOLO format with a board-level split
python src/prepare_data.py --src data/raw/PCB_DATASET --dst data/yolo
```

## Dataset
[PKU PCB Defects](https://www.kaggle.com/datasets/akhatova/pcb-defects) — 693 images, 6 defect classes
(missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper), synthetically added to 10 base boards.

## Split (by base board, no leakage)
| Split | Boards | Images | Objects |
|---|---|---|---|
| train | 01, 04, 05, 06, 07, 08 | 481 | 1880 |
| val   | 09, 11 | 120 | 608 |
| test  | 10, 12 | 92 | 465 |

Images derived from the same base board share layout and appearance. A random image-level split would put
the same board in both train and test and inflate scores, so the test boards are never seen during training.
