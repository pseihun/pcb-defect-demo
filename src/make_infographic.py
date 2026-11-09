"""One-page infographic of the `short` class error analysis.

Reads the outputs of src/analyze_errors.py (report.json + crop montages).

Usage (from the project root):
    python src/make_infographic.py                     # analysis/ -> analysis/short_error_analysis.png
    python src/make_infographic.py <analysis dir> <output png>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, gridspec
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.image as mpimg

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "analysis")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else SRC / "short_error_analysis.png")
rep = json.loads((SRC / "report.json").read_text(encoding="utf-8"))
rfp = rep["recurring_fp"]
ap_all, ap_ign = rfp["AP50_all (own calc)"], rfp["AP50_ignore_region (own calc)"]
iou_s = rep["tp_iou_median"]["short"]
iou_o = [v for k, v in rep["tp_iou_median"].items() if k != "short"]

for f in ("Malgun Gothic", "맑은 고딕"):
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#c9c8c2"
GOOD, WARN, CRIT = "#0ca30c", "#c98500", "#d03b3b"

CLASSES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
val = [rep["per_class_val"][c]["AP50"] for c in CLASSES]
test = [rep["per_class_test"][c]["AP50"] for c in CLASSES]
fp = [rep["fp_breakdown_test"][c]["conf>=0.5"]["FP"] for c in CLASSES]
st = rep["per_class_test"]["short"]
sfp = rep["fp_breakdown_test"]["short"]["conf>=0.5"]

fig = plt.figure(figsize=(12, 20), dpi=150, facecolor=SURF)
gs = gridspec.GridSpec(7, 2, figure=fig, height_ratios=[0.9, 1.05, 2.5, 5.0, 1.5, 2.2, 1.4],
                       hspace=0.5, wspace=0.42, left=0.06, right=0.96, top=0.97, bottom=0.02)

def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=10, length=0)

def section(ax, num, title, sub=None):
    ax.text(0, 1.13 if sub else 1.06, f"{num}  {title}", transform=ax.transAxes, fontsize=14,
            fontweight="bold", color=INK, va="bottom")
    if sub:
        ax.text(0, 1.04, sub, transform=ax.transAxes, fontsize=10, color=INK2, va="bottom")

# ---- Title
ax = fig.add_subplot(gs[0, :]); ax.axis("off")
ax.text(0, 0.95, "PCB 결함 탐지 · short 클래스 오류 분석", fontsize=24, fontweight="bold", color=INK, va="top")
ax.text(0, 0.38, "short가 약한 이유: 보드 12에 원래 설계된 정상 배선 하나를, 테스트 이미지마다 반복해서 short로 착각했다",
        fontsize=12.5, color=INK2, va="top")
ax.text(0, -0.05, "YOLO11s · imgsz 1280 · 테스트 = 학습에 쓰지 않은 보드 10, 12 (이미지 92장) · IoU 0.5 기준",
        fontsize=9.5, color=MUTED, va="top")

# ---- Hero numbers
ax = fig.add_subplot(gs[1, :]); ax.axis("off"); ax.set_xlim(0, 5); ax.set_ylim(0, 1)
tiles = [
    (f"{st['AP50']:.3f}", "short AP50", "다른 5개 클래스는 0.977 이상"),
    (f"{st['P']:.3f}", "정밀도 (Precision)", "short라고 한 것의 절반이 가짜"),
    (f"{st['R']:.3f}", "재현율 (Recall)", "실제 short는 대부분 찾아냄"),
    (f"{sfp['FP_by_board'].get('12', 0)} / {sfp['FP']}", "보드 12에서 나온 오탐", "confidence 0.5 이상 기준"),
    (f"{ap_all:.2f}→{ap_ign:.2f}", "그 배선 위치만 제외하면", "short AP50 (직접 계산)"),
]
for i, (big, lab, note) in enumerate(tiles):
    ax.add_patch(FancyBboxPatch((i + 0.03, 0.02), 0.94, 0.96, boxstyle="round,pad=0,rounding_size=0.06",
                                fc="#eaf2fc" if i == 4 else "#f3f2ee", ec="none"))
    ax.text(i + 0.1, 0.66, big, fontsize=21, fontweight="bold", color=INK, va="center")
    ax.text(i + 0.1, 0.33, lab, fontsize=10.5, color=INK, va="center")
    ax.text(i + 0.1, 0.13, note, fontsize=8.8, color=MUTED, va="center")

# ---- Chart 1: val vs test AP50 (dumbbell)
ax = fig.add_subplot(gs[2, 0]); style(ax)
p = ax.get_position(); ax.set_position([p.x0 + 0.07, p.y0, p.width - 0.07, p.height])
section(ax, "①", "short만 테스트 보드에서 무너진다", "클래스별 AP50: 검증 보드(09, 11) → 테스트 보드(10, 12)")
y = list(range(len(CLASSES)))[::-1]
for yi, v, t, c in zip(y, val, test, CLASSES):
    ax.plot([v, t], [yi, yi], color=GRAY, lw=2, zorder=1)
    ax.scatter(v, yi, s=70, color=BLUE, zorder=3, ec=SURF, lw=2)
    ax.scatter(t, yi, s=70, color=ORANGE, zorder=3, ec=SURF, lw=2)
    if c == "short":
        ax.text(t - 0.012, yi + 0.33, f"{t:.3f}", ha="center", fontsize=10, fontweight="bold", color=INK)
        ax.text(v, yi + 0.33, f"{v:.3f}", ha="center", fontsize=10, color=INK2)
ax.set_yticks(y); ax.set_yticklabels(CLASSES, fontsize=10.5, color=INK)
for lbl in ax.get_yticklabels():
    if lbl.get_text() == "short":
        lbl.set_fontweight("bold")
ax.set_xlim(0.6, 1.02); ax.set_ylim(-0.6, len(CLASSES) - 0.2)
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.scatter([], [], s=60, color=BLUE, label="검증 보드"); ax.scatter([], [], s=60, color=ORANGE, label="테스트 보드")
ax.legend(loc="lower left", frameon=False, fontsize=9.5, labelcolor=INK2, handletextpad=0.3)

# ---- Chart 2: FP count per class
ax = fig.add_subplot(gs[2, 1]); style(ax)
p = ax.get_position(); ax.set_position([p.x0 + 0.06, p.y0, p.width - 0.06, p.height])
section(ax, "②", "오탐은 short에만 몰려 있다", "테스트 보드 오탐 개수 (confidence ≥ 0.5)")
cols = [ORANGE if c == "short" else GRAY for c in CLASSES]
ax.barh(y, fp, color=cols, height=0.55)
for yi, v, c in zip(y, fp, CLASSES):
    ax.text(v + 1, yi, str(v), va="center", fontsize=10.5, color=INK, fontweight="bold" if c == "short" else None)
ax.set_yticks(y); ax.set_yticklabels(CLASSES, fontsize=10.5, color=INK)
ax.set_xlim(0, 70); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)

# ---- Images: FP vs TP
fpm = mpimg.imread(str(SRC / "short_top_fp.jpg")); tpm = mpimg.imread(str(SRC / "short_top_tp.jpg"))
def tile(m, r, c):
    return m[r * 240:(r + 1) * 240, c * 240:(c + 1) * 240]
outer = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[3, :], height_ratios=[0.17, 1], hspace=0.0)
hdr = fig.add_subplot(outer[0]); hdr.axis("off")
hdr.text(0, 0.78, "③  왜 헷갈렸나: 생김새가 거의 같다", transform=hdr.transAxes, fontsize=14,
         fontweight="bold", color=INK, va="bottom")
hdr.text(0, 0.5, "빨간 박스 = 모델이 short라고 예측한 위치 · 초록 박스 = 실제 short를 맞힌 위치 (각 사진 좌상단: 보드 번호, confidence)",
         transform=hdr.transAxes, fontsize=10, color=INK2, va="bottom")
sub = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=outer[1], hspace=0.16, wspace=0.05,
                                       width_ratios=[1, 1, 1, 1.15])
rows = [("오탐 · 보드 12의 정상 배선", [tile(fpm, 0, 1), tile(fpm, 0, 2), tile(fpm, 1, 0)], CRIT,
         f"두 핀을 잇도록 원래 설계된 배선.\n보드 12 이미지 {rfp['bg_fp_in_region_conf>=0.25']}장 모두에서\n같은 자리를 short로 잡음.\n"
         "confidence 약 0.69로, 진짜 short(0.7대)와\n거의 같아서 순위가 섞임 → AP 급락."),
        ("정탐 · 실제 short 결함", [tile(tpm, 0, 0), tile(tpm, 0, 2), tile(tpm, 1, 1)], GOOD,
         "나란히 있는 두 선 사이를\n구리가 'H'자로 잘못 이어버린 결함.\n\n두 모양 모두 '구리로 이어진 두 도체'.")]
for r, (title, ims, col, desc) in enumerate(rows):
    for c, im in enumerate(ims):
        a = fig.add_subplot(sub[r, c]); a.imshow(im); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_color(col); s.set_linewidth(2.5)
        if c == 0:
            a.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK, pad=5)
    a = fig.add_subplot(sub[r, 3]); a.axis("off")
    a.text(0.04, 0.5, desc, fontsize=10, color=INK2, va="center", linespacing=1.55)

# ---- Short FP breakdown (stacked bars)
ax = fig.add_subplot(gs[4, :]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
section(ax, "④", f"short 오탐 {sfp['FP']}개 해부 (confidence ≥ 0.5)")
def bar(y0, label, parts):
    ax.text(0, y0 + 0.07, label, fontsize=10.5, color=INK, va="center")
    total = sum(v for _, v, _ in parts); x = 0.2; W = 0.78
    for name, v, col in parts:
        w = W * v / total
        if v:
            ax.add_patch(Rectangle((x, y0), max(w - 0.003, 0.002), 0.14, fc=col, ec="none"))
        x += w
    xs = 0.2
    for name, v, col in parts:
        ax.text(xs if v else xs, y0 - 0.1, f"{name} {v}", fontsize=9.5, color=INK2 if v else MUTED, va="center")
        xs += 0.2
t = sfp["FP_by_type"]
bar(0.55, "오탐 유형", [("배경 오탐 (결함 없는 곳)", t.get("background", 0), ORANGE),
                       ("위치 오차", t.get("localization", 0), BLUE),
                       ("다른 결함과 혼동", t.get("class_confusion", 0), GRAY)])
b = sfp["FP_by_board"]
bar(0.12, "보드", [("보드 12", b.get("12", 0), ORANGE), ("보드 10", b.get("10", 0), BLUE)])

# ---- Hypothesis table
ax = fig.add_subplot(gs[5, :]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
section(ax, "⑤", "가설 검증 결과")
hyp = [
    ("확인", GOOD, "처음 보는 보드 패턴을 short로 착각",
     f"conf ≥ 0.5 오탐 {sfp['FP']}개 전부가 보드 12의 한 위치. 그 위치 제외 시 AP50 {ap_all:.2f} → {ap_ign:.2f}"),
    ("일부", WARN, "박스 위치 오차",
     f"IoU 0.5 문턱에서 놓친 것 {rep['fp_breakdown_test']['short']['conf>=0.25']['FP_by_type'].get('localization', 0)}건"
     f"(conf ≥ 0.25). 박스 IoU 중앙값 {iou_s:.2f} "
     f"(다른 클래스 {min(iou_o):.2f}~{max(iou_o):.2f}) → AP50-95 {st['AP50_95']:.3f}"),
    ("기각", CRIT, "비슷한 결함(spur 등)과 혼동", "다른 결함과 헷갈린 오탐 0건"),
    ("기각", CRIT, "학습 데이터 부족", f"학습 short {rep['label_stats']['train']['short']['count']}개로 타 클래스와 비슷, 검증 보드 AP50 {rep['per_class_val']['short']['AP50']:.3f}"),
    ("기각", CRIT, "테스트 라벨 누락", "오탐 위치는 결함이 아니라 원래 설계된 배선"),
]
rh = 0.17
for i, (verdict, col, h, ev) in enumerate(hyp):
    yy = 0.9 - (i + 1) * rh
    if i % 2 == 0:
        ax.add_patch(Rectangle((0, yy), 1, rh, fc="#f3f2ee", ec="none"))
    ax.add_patch(FancyBboxPatch((0.012, yy + rh * 0.22), 0.062, rh * 0.56, boxstyle="round,pad=0,rounding_size=0.01",
                                fc=col, ec="none"))
    ax.text(0.043, yy + rh / 2, verdict, ha="center", va="center", fontsize=10, fontweight="bold", color="white")
    ax.text(0.09, yy + rh / 2, h, va="center", fontsize=10.5, color=INK, fontweight="bold")
    ax.text(0.37, yy + rh / 2, ev, va="center", fontsize=9.8, color=INK2)

# ---- Conclusion
ax = fig.add_subplot(gs[6, :]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.04", fc="#eaf2fc", ec="none"))
ax.text(0.025, 0.8, "결론", fontsize=13, fontweight="bold", color=INK, va="center")
ax.text(0.025, 0.5, "모델이 short를 못 배운 것이 아니다. 작은 영역의 생김새만으로는 '의도된 연결'과 '실수로 생긴 연결'을 구분할 수 없는,\n"
        "외관 기반 탐지기의 구조적 한계다. 실제 PCB 자동 검사(AOI)가 기준 보드나 설계 데이터와 비교하는 이유이기도 하다.",
        fontsize=10.5, color=INK, va="center", linespacing=1.6)
ax.text(0.025, 0.15, "다음 단계: 결함 없는 원본 보드(PCB_USED)와 비교하는 방식 실험 · 정상 배선을 '결함 없음' 예시로 학습에 추가",
        fontsize=9.5, color=INK2, va="center")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, facecolor=SURF)
print("saved", OUT)
