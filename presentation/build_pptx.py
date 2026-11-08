"""Generate draft deck v2 (structure review, before the evidence phase).

Outline: `pm/PLAN.md` (Deck outline). Review checklist it answers:
`pm/research/deck-review-2026-11-07.md`. Slides whose evidence does not exist
yet (T05 A1-A10, T06, T07, T08) carry a labelled placeholder box where the
figure or table will go, so the narrative can be reviewed now.

Every number on a slide or in the notes is READ from tracked files at build
time and formatted through `Results.num` (or `Results.illus` for the clearly
labelled illustrative examples), which records it. `tests/test_build_pptx.py`
scans every slide and every note for numerals and fails on any that did not
pass through those formatters, so a hand-typed number cannot reach the deck.

Sources
  results/summary.csv, results/ablation.csv, results/ablation_gate.csv
  results/analysis/reliability_stats.csv, results/analysis/dataset_constants.csv
      (written by `analysis/deck_evidence.py` from results/raw/**.json, configs/)
  results/analysis/literature_values.csv  (published numbers, with their source)
  results/raw/ablation/*_margin.json       (admission rates)
  configs/method/ablation/margin.yaml, configs/method/cal_tda_gate.yaml
  utils/loo_temperature.default_grid       (the temperature grid)

    python analysis/deck_evidence.py && python presentation/build_pptx.py

Writes `presentation/calibration_aware_tta.pptx` (16:9) and the generated
speaker-notes mirror `presentation/slides.md`.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
from pathlib import Path

import pandas as pd
import yaml
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from analysis.aggregate import DATASET_LABEL, DATASET_ORDER  # noqa: E402
from analysis.deck_evidence import flat_cache_top_probability  # noqa: E402
from analysis.make_contribution_tables import _admission_rates  # noqa: E402

ROOT = _ROOT
ASSETS = ROOT / "presentation" / "assets"
FIGURES = ROOT / "analysis" / "figures"

# --- Theme (same palette as deck v1 and the figures) ---------------------------
INK = RGBColor(0x1A, 0x1F, 0x2B)
MUTED = RGBColor(0x55, 0x5C, 0x6B)
ACCENT = RGBColor(0x7D, 0x5B, 0xA6)       # ours
BASELINE = RGBColor(0xDD, 0x84, 0x52)     # TDA
CLIPBLUE = RGBColor(0x4C, 0x72, 0xB0)     # CLIP
NEUTRAL = RGBColor(0xC9, 0xC6, 0xC0)
RULE = RGBColor(0xD8, 0xD5, 0xD0)
SURFACE = RGBColor(0xFF, 0xFF, 0xFF)
WARN = RGBColor(0xB3, 0x4A, 0x3C)
GOOD = RGBColor(0x3C, 0x8C, 0x5A)
BG = RGBColor(0xF4, 0xF0, 0xE9)
BG_DARK = RGBColor(0x25, 0x1F, 0x36)
HEADER_FILL = RGBColor(0xE7, 0xE2, 0xD8)
TINT_ACCENT = RGBColor(0xEC, 0xE6, 0xF3)
TINT_WARN = RGBColor(0xF7, 0xEA, 0xE1)
TINT_CLIP = RGBColor(0xE4, 0xEB, 0xF5)
PENDING_FILL = RGBColor(0xFF, 0xF6, 0xD9)
PENDING_LINE = RGBColor(0xB8, 0x8A, 0x1B)

FONT = "Calibri"
BODY = 20          # main bullets
SUB = 18           # sub-bullets, panel text: the floor for body text
TABLE = 14         # table text floor
SMALL = 14         # source lines and figure captions only

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
MARGIN = Inches(0.6)
BODY_TOP = Inches(1.62)
BODY_BOTTOM = Inches(6.85)
CONTENT_W = SLIDE_W - 2 * MARGIN

# Numerals recognised as data on a slide: a digit not glued to a word, a slash,
# an apostrophe or a hyphen (so "ViT-B/16", "CVPR'24", "A3", "top-2", "16-shot"
# and "RN50" are names, not numbers). The test uses the same pattern.
NUMBER_RE = re.compile(r"(?<![\w'/.\-])\d[\d,]*(?:\.\d+)?(?![\w'/\-])")


def number_tokens(text: str) -> list[str]:
    return [token.rstrip(",") for token in NUMBER_RE.findall(text)]


# Inputs of the illustrative examples. These are NOT results: every slide that
# uses one says "Illustrative" on it, and the test checks that it does.
ILLUSTRATIVE = {
    "toy_p": (0.60, 0.30, 0.10),              # CLIP's distribution, 3 classes
    "toy_q_agree": (0.60, 0.30, 0.10),        # a cache that agrees
    "toy_q_disagree": (0.10, 0.30, 0.60),     # a cache that disagrees
    "we_classes": ("Birman", "Ragdoll", "Siamese"),
    "we_text_cos": (0.300, 0.290, 0.250),     # cosine of x to each class prompt
    "we_cache_cos": ((0.90, 0.85), (0.80,), ()),  # cosines of x to cached items, per class
    "scale_cos": 0.80,                        # a typical cached-neighbour cosine
    "peaked_clip_top": 0.90,                  # a confident CLIP top-1 probability
    "sketch_clear": (0.84, 0.07, 0.04, 0.03, 0.02),
    "sketch_tie": (0.44, 0.37, 0.09, 0.06, 0.04),
    "sketch_spread": (0.26, 0.21, 0.20, 0.17, 0.16),
}


# --- Data ----------------------------------------------------------------------
class Results:
    """Every number the deck shows, read from tracked files, formatted and recorded."""

    def __init__(self, root: Path = ROOT):
        root = Path(root)
        self.summary = pd.read_csv(root / "results/summary.csv")
        self.ablation = pd.read_csv(root / "results/ablation.csv")
        self.ablation_gate = pd.read_csv(root / "results/ablation_gate.csv")
        self.rel = pd.read_csv(root / "results/analysis/reliability_stats.csv")
        self.const = pd.read_csv(root / "results/analysis/dataset_constants.csv").set_index("dataset")
        self.lit = pd.read_csv(root / "results/analysis/literature_values.csv").set_index("key")
        self.admission = _admission_rates(root / "results/raw/ablation")
        self.margin_cfg = yaml.safe_load((root / "configs/method/ablation/margin.yaml").read_text(encoding="utf-8"))
        self.gate_cfg = yaml.safe_load((root / "configs/method/cal_tda_gate.yaml").read_text(encoding="utf-8"))
        from utils.loo_temperature import default_grid
        self.grid = default_grid()
        self.emitted: set[str] = set()
        self.illustrative: set[str] = set()

    # formatting -------------------------------------------------------------
    def _record(self, text: str, bucket: set) -> str:
        bucket.update(number_tokens(text))
        return text

    def num(self, value, digits=2, *, sign=False, comma=False) -> str:
        """Format a value read from data and record it as legitimately on the deck."""
        if comma:
            text = f"{value:,.0f}"
        elif sign:
            text = f"{value:+.{digits}f}".replace("-", "−")
        else:
            text = f"{value:.{digits}f}"
            if value < 0:
                text = text.replace("-", "−")
        return self._record(text, self.emitted)

    def illus(self, value, digits=2) -> str:
        """Format an illustrative (non-result) value; its slide must say so."""
        text = f"{value:.{digits}f}"
        return self._record(text, self.illustrative)

    # lookups ----------------------------------------------------------------
    def s(self, dataset, method, column):
        row = self.summary[(self.summary.dataset == dataset) & (self.summary.method == method)]
        if row.empty:
            raise KeyError((dataset, method))
        return float(row.iloc[0][column])

    def s_mean(self, method, column):
        return float(self.summary[self.summary.method == method][column].mean())

    def r(self, study, dataset, variant, column):
        row = self.rel[(self.rel.study == study) & (self.rel.dataset == dataset) & (self.rel.variant == variant)]
        if row.empty:
            raise KeyError((study, dataset, variant))
        return float(row.iloc[0][column])

    def r_mean(self, study, variant, column):
        frame = self.rel[(self.rel.study == study) & (self.rel.variant == variant)]
        return float(frame[column].mean())

    def v(self, dataset, variant, column, study="ablation"):
        frame = self.ablation if study == "ablation" else self.ablation_gate
        row = frame[(frame.dataset == dataset) & (frame.variant == variant)]
        if row.empty:
            raise KeyError((study, dataset, variant))
        return float(row.iloc[0][column])

    def v_mean(self, variant, column, study="ablation"):
        frame = self.ablation if study == "ablation" else self.ablation_gate
        return float(frame[frame.variant == variant][column].mean())

    def c(self, dataset, column):
        return self.const.loc[dataset, column]

    def litv(self, key):
        return float(self.lit.loc[key, "value"])

    def n_bins(self):
        values = set(self.rel["n_bins"])
        if len(values) != 1:
            raise ValueError(f"records disagree on the number of ECE bins: {values}")
        return int(values.pop())

    def total_images(self):
        return int(self.summary.groupby("dataset")["n_samples"].first().sum())


# --- Low-level drawing helpers ---------------------------------------------------
def _style(run, size, *, bold=False, color=INK, italic=False, font=FONT):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = font


def textbox(slide, left, top, width, height, *, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(int(left), int(top), int(width), int(height))
    frame = box.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = anchor
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0
    return box, frame


def add_text(slide, left, top, width, height, paragraphs, *, anchor=MSO_ANCHOR.TOP,
             align=PP_ALIGN.LEFT, spacing=6):
    """`paragraphs`: list of (text, size, bold, color) or lists of such runs."""
    _, frame = textbox(slide, left, top, width, height, anchor=anchor)
    for index, para in enumerate(paragraphs):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.alignment = align
        if index:
            paragraph.space_before = Pt(spacing)
        runs = para if isinstance(para, list) else [para]
        for text, size, bold, color in runs:
            run = paragraph.add_run()
            run.text = text
            _style(run, size, bold=bold, color=color)
    return frame


def paint_background(slide, color=BG):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color
    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, Inches(0.09))
    band.fill.solid()
    band.fill.fore_color.rgb = ACCENT
    band.line.fill.background()
    band.shadow.inherit = False


def new_slide(prs, title, *, kicker=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    paint_background(slide)
    top = Inches(0.36)
    if kicker:
        _, frame = textbox(slide, MARGIN, top, CONTENT_W, Inches(0.3))
        run = frame.paragraphs[0].add_run()
        run.text = kicker.upper()
        _style(run, 14, bold=True, color=ACCENT)
        top += Inches(0.34)
    box, frame = textbox(slide, MARGIN, top, CONTENT_W, Inches(0.62))
    box.name = "title"
    run = frame.paragraphs[0].add_run()
    run.text = title
    _style(run, 30, bold=True, color=INK)
    return slide


def add_bullets(slide, items, *, left=None, top=None, width=None, height=None,
                size=BODY, sub_size=SUB, spacing=8):
    """`items`: strings, (level, text) or (level, text, color)."""
    left = MARGIN if left is None else left
    top = BODY_TOP if top is None else top
    width = CONTENT_W if width is None else width
    height = (BODY_BOTTOM - top) if height is None else height
    _, frame = textbox(slide, left, top, width, height)
    first = True
    for item in items:
        if isinstance(item, str):
            level, text, color = 0, item, INK
        elif len(item) == 2:
            (level, text), color = item, INK
        else:
            level, text, color = item
        paragraph = frame.paragraphs[0] if first else frame.add_paragraph()
        first = False
        paragraph.space_after = Pt(spacing if level == 0 else spacing * 0.6)
        if not text:
            _style(paragraph.add_run(), 8)
            continue
        hang = Inches(0.30)
        properties = paragraph._p.get_or_add_pPr()
        properties.set("marL", str(int(hang if level == 0 else hang + Inches(0.42))))
        properties.set("indent", str(int(-hang)))
        run = paragraph.add_run()
        run.text = ("•   " if level == 0 else "–   ") + text
        _style(run, size if level == 0 else sub_size, color=color if level == 0 else MUTED)
    return frame


def add_panel(slide, left, top, width, height, *, fill=SURFACE, line=RULE, dashed=False):
    panel = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, int(left), int(top), int(width), int(height))
    panel.fill.solid()
    panel.fill.fore_color.rgb = fill
    if line is None:
        panel.line.fill.background()
    else:
        panel.line.color.rgb = line
        panel.line.width = Pt(1.5 if dashed else 1.25)
        if dashed:
            panel.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    panel.shadow.inherit = False
    panel.adjustments[0] = 0.05
    panel.text_frame.text = ""
    return panel


def add_callout(slide, text, *, top=Inches(6.18), color=ACCENT, size=SUB, left=None, width=None,
                height=Inches(0.62)):
    left = MARGIN if left is None else left
    width = CONTENT_W if width is None else width
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, int(left), int(top), Pt(5), int(height))
    bar.fill.solid()
    bar.fill.fore_color.rgb = color
    bar.line.fill.background()
    bar.shadow.inherit = False
    _, frame = textbox(slide, left + Inches(0.2), top, width - Inches(0.2), height, anchor=MSO_ANCHOR.MIDDLE)
    run = frame.paragraphs[0].add_run()
    run.text = text
    _style(run, size, bold=True, color=INK)
    return frame


def add_placeholder(slide, left, top, width, height, evidence, what, line):
    """A clearly styled box where pending evidence will go."""
    add_panel(slide, left, top, width, height, fill=PENDING_FILL, line=PENDING_LINE, dashed=True)
    pad = Inches(0.18)
    add_text(slide, left + pad, top + pad, width - 2 * pad, height - 2 * pad, [
        (f"[Pending {evidence}: {what}]", SUB, True, PENDING_LINE),
        (line, SUB, False, INK),
    ], anchor=MSO_ANCHOR.MIDDLE, spacing=4)


def add_picture(slide, path, left, top, max_width, max_height, *, align="center"):
    picture = slide.shapes.add_picture(str(path), int(left), int(top), width=int(max_width))
    if picture.height > max_height:
        picture._element.getparent().remove(picture._element)
        picture = slide.shapes.add_picture(str(path), int(left), int(top), height=int(max_height))
    if align == "center":
        picture.left = int(left + (max_width - picture.width) / 2)
    picture.top = int(top + (max_height - picture.height) / 2)
    return picture


def add_table(slide, header, rows, left, top, width, *, col_widths=None, font_size=TABLE,
              row_height=Inches(0.4), highlight_rows=(), first_col_align=PP_ALIGN.LEFT,
              align=PP_ALIGN.RIGHT, bold_first_col=False, cell_colors=None, aligns=None):
    """Text columns can be left-aligned via `aligns` ("L"/"R" per column)."""
    def column_align(col):
        if aligns:
            return PP_ALIGN.LEFT if aligns[col] == "L" else PP_ALIGN.RIGHT
        return first_col_align if col == 0 else align

    shape = slide.shapes.add_table(len(rows) + 1, len(header), int(left), int(top), int(width),
                                   int(row_height * (len(rows) + 1)))
    table = shape.table
    table.first_row = True
    table.horz_banding = False
    if col_widths:
        total = sum(col_widths)
        for index, fraction in enumerate(col_widths):
            table.columns[index].width = Emu(int(width * fraction / total))
    for col, text in enumerate(header):
        cell = table.cell(0, col)
        cell.text = ""
        cell.fill.solid()
        cell.fill.fore_color.rgb = HEADER_FILL
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_left = cell.margin_right = Inches(0.07)
        cell.margin_top = cell.margin_bottom = Inches(0.03)
        paragraph = cell.text_frame.paragraphs[0]
        paragraph.alignment = column_align(col)
        run = paragraph.add_run()
        run.text = text
        _style(run, font_size, bold=True, color=MUTED)
    table.rows[0].height = int(row_height)
    for r_index, row in enumerate(rows, start=1):
        table.rows[r_index].height = int(row_height)
        emphasised = (r_index - 1) in highlight_rows
        for col, text in enumerate(row):
            cell = table.cell(r_index, col)
            cell.text = ""
            cell.fill.solid()
            cell.fill.fore_color.rgb = TINT_ACCENT if emphasised else SURFACE
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = cell.margin_right = Inches(0.07)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            paragraph = cell.text_frame.paragraphs[0]
            paragraph.alignment = column_align(col)
            run = paragraph.add_run()
            run.text = text
            color = INK
            if cell_colors and (r_index - 1, col) in cell_colors:
                color = cell_colors[(r_index - 1, col)]
            _style(run, font_size, bold=emphasised or (bold_first_col and col == 0), color=color)
    return table


def flow_box(slide, left, top, width, height, title, subtitle=None, *, fill=SURFACE, border=RULE,
             title_size=SUB, subtitle_size=SUB, title_color=INK):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, int(left), int(top), int(width), int(height))
    box.fill.solid()
    box.fill.fore_color.rgb = fill
    box.line.color.rgb = border
    box.line.width = Pt(1.5)
    box.shadow.inherit = False
    box.adjustments[0] = 0.08
    frame = box.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    frame.margin_left = frame.margin_right = Inches(0.1)
    frame.margin_top = frame.margin_bottom = Inches(0.05)
    paragraph = frame.paragraphs[0]
    paragraph.alignment = PP_ALIGN.CENTER
    run = paragraph.add_run()
    run.text = title
    _style(run, title_size, bold=True, color=title_color)
    if subtitle:
        paragraph = frame.add_paragraph()
        paragraph.alignment = PP_ALIGN.CENTER
        paragraph.space_before = Pt(4)
        run = paragraph.add_run()
        run.text = subtitle
        _style(run, subtitle_size, color=MUTED)
    return box


def arrow(slide, left, top, width, height, shape=MSO_SHAPE.RIGHT_ARROW, color=NEUTRAL):
    a = slide.shapes.add_shape(shape, int(left), int(top), int(width), int(height))
    a.fill.solid()
    a.fill.fore_color.rgb = color
    a.line.fill.background()
    a.shadow.inherit = False
    a.text_frame.text = ""
    return a


def dot(slide, cx, cy, d, color, *, outline=None, shape=MSO_SHAPE.OVAL):
    s = slide.shapes.add_shape(shape, int(cx - d / 2), int(cy - d / 2), int(d), int(d))
    s.fill.solid()
    s.fill.fore_color.rgb = color
    if outline is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = outline
        s.line.width = Pt(2.25)
    s.shadow.inherit = False
    s.text_frame.text = ""
    return s


def add_footer(slide, index, total, label=None):
    box, frame = textbox(slide, MARGIN, Inches(7.06), Inches(8.0), Inches(0.3))
    box.name = "footer"
    run = frame.paragraphs[0].add_run()
    run.text = "Calibration-aware TTA for CLIP  ·  draft v2 for review"
    _style(run, 12, color=MUTED)
    box, frame = textbox(slide, SLIDE_W - MARGIN - Inches(1.4), Inches(7.06), Inches(1.4), Inches(0.3))
    box.name = "footer"
    frame.paragraphs[0].alignment = PP_ALIGN.RIGHT
    run = frame.paragraphs[0].add_run()
    run.text = label or f"{index} / {total}"
    _style(run, 12, color=MUTED)


def set_notes(slide, text):
    """Notes as paragraphs: blank lines separate paragraphs, single newlines are rewrapped."""
    paragraphs = [" ".join(p.split()) for p in text.strip().split("\n\n") if p.strip()]
    slide.notes_slide.notes_text_frame.text = "\n\n".join(paragraphs)


# --- Rendered assets (equations and small illustrative figures) -----------------
def equation(tex: str, *, size=24, color="#1A1F2B") -> Path:
    """Render a mathtext equation to a transparent PNG at its true point size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = hashlib.sha1(f"{tex}|{size}|{color}".encode()).hexdigest()[:12]
    path = ASSETS / f"eq_{key}.png"
    if path.exists():
        return path
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${tex}$", fontsize=size, color=color)
    fig.savefig(path, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return path


def add_equation(slide, tex, left, top, *, size=24, color="#1A1F2B", center_width=None):
    path = equation(tex, size=size, color=color)
    from PIL import Image
    with Image.open(path) as image:
        w_px, h_px = image.size
    width, height = Inches(w_px / 300), Inches(h_px / 300)
    if center_width is not None:
        if width > center_width:   # never let an equation overflow its column
            height = int(height * center_width / width)
            width = center_width
        left = left + (center_width - width) / 2
    slide.shapes.add_picture(str(path), int(left), int(top), width=int(width), height=int(height))
    return height


def render_vote_figure(results: Results, out_path: Path) -> Path:
    """Positive-cache vote weight exp(-beta (1 - cos)) for each dataset's beta."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = ["#C44E52", "#937860", "#DA8BC3", "#4D4D4D", "#17BECF"]
    cos = np.linspace(0.5, 1.0, 200)
    fig, ax = plt.subplots(figsize=(5.4, 4.1))
    for color, dataset in zip(colors, DATASET_ORDER):
        beta = float(results.c(dataset, "beta"))
        ax.plot(cos, np.exp(-beta * (1 - cos)), color=color, linewidth=2.4,
                label=f"{DATASET_LABEL[dataset]} (β = {beta:g})")
    ax.axhline(0.5, color="#999999", linewidth=1, linestyle=":")
    ax.set_xlabel("cosine between test and cached feature", fontsize=14)
    ax.set_ylabel("vote weight", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=12, frameon=False, loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def render_calibration_figure(out_path: Path) -> Path:
    """Textbook reliability diagram from SYNTHETIC curves (concept only)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    confidence = np.linspace(0.30, 0.975, 14)
    over = confidence ** 2.4
    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#4D4D4D", linewidth=1.8, label="calibrated")
    ax.plot(confidence, over, marker="o", markersize=6, color="#B34A3C", linewidth=2.2,
            label="overconfident")
    pick = 10
    ax.plot([confidence[pick]] * 2, [over[pick], confidence[pick]], color="#7D5BA6", linewidth=3.2,
            label="one bin's gap")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("confidence (bin mean)", fontsize=14)
    ax.set_ylabel("accuracy in the bin", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=13, frameon=True, framealpha=0.95, edgecolor="#D8D5D0")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def softmax(values):
    m = max(values)
    e = [math.exp(v - m) for v in values]
    s = sum(e)
    return [x / s for x in e]


def product_of_experts(p, q, alpha):
    w = [pi * qi ** alpha for pi, qi in zip(p, q)]
    s = sum(w)
    return [x / s for x in w]


def mixture(p, q, w=0.5):
    return [(1 - w) * pi + w * qi for pi, qi in zip(p, q)]


# --- Slides -----------------------------------------------------------------------
# Each builder: (prs, r) -> slide. Notes are spoken English; numbers via r.num.

def slide_title(prs, r):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = BG_DARK
    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, SLIDE_H - Inches(0.14), SLIDE_W, Inches(0.14))
    band.fill.solid()
    band.fill.fore_color.rgb = ACCENT
    band.line.fill.background()
    light, lavender, soft = RGBColor(0xF5, 0xF2, 0xFA), RGBColor(0xB9, 0xA6, 0xD6), RGBColor(0xC8, 0xC2, 0xD8)
    add_text(slide, Inches(1.1), Inches(1.6), Inches(11.2), Inches(0.4),
             [("TAU  ·  DEEP LEARNING  ·  FINAL PROJECT  ·  DRAFT v2 FOR REVIEW", 16, True, lavender)])
    add_text(slide, Inches(1.1), Inches(2.15), Inches(11.2), Inches(1.8),
             [("Calibration-Aware Test-Time Adaptation for CLIP", 44, True, light)])
    add_text(slide, Inches(1.1), Inches(4.25), Inches(11.2), Inches(1.6), [
        ("Gal Machlin   ·   Idan Yehuda", 22, False, light),
        ("Zero-shot CLIP, an unlabeled test stream, no backprop: can test-time adaptation "
         "keep its accuracy gain and report honest confidence?", 20, False, soft),
    ], spacing=14)
    add_text(slide, Inches(1.1), Inches(6.35), Inches(11.2), Inches(0.5), [
        ("Yellow dashed boxes mark evidence that will be added after feature extraction.", 16, False, lavender),
    ])
    set_notes(slide, """
Hello everyone. This project is about test-time adaptation for CLIP, and specifically
about whether the confidence such methods report can be trusted.

The setting is strict: we only know the class names, the test images arrive one at a time
without labels, and nothing is retrained. Within that setting, a well-known method called
TDA improves accuracy but makes the model overconfident. We study why, propose four small
changes, and show which ones work.

This is a draft for structure review. Wherever you see a yellow dashed box, that is a
figure or table that will be computed once the image features are extracted.
""")
    return slide


def slide_agenda(prs, r):
    slide = new_slide(prs, "Roadmap", kicker="Overview")
    entries = [
        ("1", "Background", "the zero-shot setting, CLIP, why it fails, the method landscape"),
        ("2", "TDA in depth", "how its cache is written and read, the math, what it buys"),
        ("3", "Calibration", "ECE, why it matters, and how TDA becomes overconfident"),
        ("4", "Our proposal", "four changes, each: motive, why it should work, evidence, result"),
        ("5", "Results", "headline table, fairness protocols, ablation, baselines"),
        ("6", "Limitations and conclusion", "what we can and cannot claim yet"),
    ]
    top = Inches(1.75)
    for index, (number, heading, detail) in enumerate(entries):
        row_top = top + index * Inches(0.82)
        chip = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, MARGIN, int(row_top), Inches(0.62), Inches(0.62))
        chip.fill.solid()
        chip.fill.fore_color.rgb = ACCENT if index in (1, 3) else HEADER_FILL
        chip.line.fill.background()
        chip.shadow.inherit = False
        chip.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = chip.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = number
        _style(run, 20, bold=True, color=SURFACE if index in (1, 3) else MUTED)
        add_text(slide, MARGIN + Inches(0.9), row_top, Inches(11.0), Inches(0.62), [[
            (heading, 22, True, INK), ("      " + detail, SUB, False, MUTED)]], anchor=MSO_ANCHOR.MIDDLE)
    set_notes(slide, """
Six parts. First the background: what exactly the task is, what CLIP does, and where TDA
sits among other methods. Then a long section on TDA itself, with the math, because
everything we do changes one specific line of it. Then calibration: what we measure and
why TDA gets it wrong. Then our four changes, one at a time, each with its motivation,
why we expected it to work, the evidence and the outcome. Then results, fairness and
baselines, and finally the limitations.
""")
    return slide


def slide_problem(prs, r):
    slide = new_slide(prs, "The problem: classify with nothing but the class names",
                      kicker="1 · Background")
    pets_classes = r.num(r.c("pets", "n_classes"), 0)
    add_bullets(slide, [
        f"Given: the names of the classes, e.g. the {pets_classes} breeds of Oxford Pets.",
        "Given: a stream of unlabeled test images, arriving one at a time.",
        "Not given: any labeled image. No training set, no labeled validation set.",
        "Not allowed: retraining or backprop. One forward pass per image, and each "
        "prediction is final before the next image arrives (online).",
        "Wanted: the right class AND an honest confidence for it.",
    ], width=Inches(7.3), spacing=12)
    add_panel(slide, Inches(8.25), Inches(1.75), Inches(4.48), Inches(4.2))
    seeds = sorted(set(r.summary["seed"]))
    backbone = sorted(set(r.summary["backbone"]))[0]
    add_text(slide, Inches(8.5), Inches(1.95), Inches(4.0), Inches(3.9), [
        ("Our evaluation", SUB, True, ACCENT),
        (f"Backbone {backbone}, single centre crop", SUB, False, INK),
        ("Prompt: one template for every class", SUB, False, INK),
        (f"Five datasets, CoOp test splits, {r.num(r.total_images(), comma=True)} test images", SUB, False, INK),
        (f"Batch size one, seed {r.num(seeds[0], 0)}", SUB, False, INK),
        (f"ECE with {r.num(r.n_bins(), 0)} equal-width bins", SUB, False, INK),
        ("Labels are read only by the final scorer", SUB, False, INK),
    ], spacing=8)
    add_callout(slide, "Same setting as the benchmark we fork (Illusion of Progress, NeurIPS'25) and as TDA.")
    set_notes(slide, f"""
Let me state the task precisely, because it rules out a lot of familiar tools.

We are given the class names, for example the {pets_classes} pet breeds, and a stream of test
images without labels. There is no labeled set at all, not even for validation. We may not
train or backpropagate; each image gets one forward pass, and its prediction is final
before the next image arrives.

And we want two things: the right class, and a confidence we can believe.

On the right is our evaluation setup: five datasets, {r.num(r.total_images(), comma=True)} test images in total, one image at a
time, and calibration measured with {r.num(r.n_bins(), 0)} bins. It is the same setting as the benchmark
paper whose code we forked.
""")
    return slide


def slide_clip(prs, r):
    slide = new_slide(prs, "Zero-shot CLIP: two frozen encoders, one prompt", kicker="1 · Background")
    scale = r.num(r.litv("clip_logit_scale"), 0)
    # diagram: image branch and text branch meeting at the logits
    top1, top2, h = Inches(1.8), Inches(3.25), Inches(1.0)
    flow_box(slide, MARGIN, top1, Inches(2.3), h, "test image x", fill=SURFACE)
    arrow(slide, MARGIN + Inches(2.38), top1 + Inches(0.34), Inches(0.5), Inches(0.32))
    flow_box(slide, MARGIN + Inches(2.95), top1, Inches(3.3), h, "image encoder f", "frozen",
             fill=TINT_CLIP, border=CLIPBLUE)
    flow_box(slide, MARGIN, top2, Inches(2.3), h, "class name k", fill=SURFACE)
    arrow(slide, MARGIN + Inches(2.38), top2 + Inches(0.34), Inches(0.5), Inches(0.32))
    flow_box(slide, MARGIN + Inches(2.95), top2, Inches(3.3), h, "text encoder g", "frozen",
             fill=TINT_CLIP, border=CLIPBLUE)
    add_text(slide, MARGIN, top2 + h + Inches(0.12), Inches(6.3), Inches(0.45),
             [("prompt:  “a photo of a {class}.”  (a single template)", SUB, True, ACCENT)])
    arrow(slide, MARGIN + Inches(6.35), top1 + Inches(1.05), Inches(0.5), Inches(0.4))
    add_panel(slide, MARGIN + Inches(7.0), top1, Inches(5.1), Inches(2.45), fill=SURFACE)
    add_equation(slide, r"z_k = s\,\cos\left(f(x),\,g(t_k)\right)", MARGIN + Inches(7.1), top1 + Inches(0.3),
                 size=26, center_width=Inches(4.9))
    add_equation(slide, r"p = \mathrm{softmax}(z),\quad \hat{y} = \arg\max_k\, z_k",
                 MARGIN + Inches(7.1), top1 + Inches(1.35), size=26, center_width=Inches(4.9))
    add_bullets(slide, [
        "Both encoders are frozen: no weight changes during test-time adaptation. Every method "
        "in this talk only changes what is done with the features and logits.",
        f"s is CLIP's learned logit scale ({scale} for this checkpoint), so logits are {scale} × a cosine.",
        "Confidence = max_k p_k, the probability of the predicted class.",
    ], top=Inches(4.75), spacing=6)
    set_notes(slide, f"""
This is the model everything builds on. CLIP has an image encoder and a text encoder.
Both are frozen throughout; nothing in this talk changes a single weight of either.

To classify, each class name is put into one prompt, "a photo of a" class, and encoded.
We use one template only, the same for every dataset. The logit for class k is the cosine
between the image feature and that text feature, times CLIP's logit scale, which is {scale}.
The softmax of the logits is the predicted distribution, the argmax is the prediction, and
its probability is what we call the confidence.

Every method later in the talk works on these features and logits; none retrains anything.
""")
    return slide


def slide_clip_fails(prs, r):
    slide = new_slide(prs, "Why zero-shot CLIP fails on these datasets", kicker="1 · Background")
    domain = {
        "dtd": "texture close-ups", "flower102": "flower species", "pets": "cat and dog breeds",
        "eurosat": "satellite land-use tiles", "aircraft": "aircraft model variants",
    }
    reason = {
        "dtd": "domain shift", "flower102": "fine-grained names", "pets": "names are common on the web",
        "eurosat": "domain shift", "aircraft": "fine-grained names",
    }
    rows = [[DATASET_LABEL[d], domain[d], r.num(r.c(d, "n_classes"), 0), r.num(r.s(d, "clipzs", "accuracy")),
             reason[d]] for d in DATASET_ORDER]
    add_table(slide, ["Dataset", "What it shows", "Classes", "CLIP top-1 (%)", "Main difficulty"], rows,
              MARGIN, Inches(1.7), CONTENT_W, col_widths=[1.4, 2.6, 1.0, 1.4, 2.6], row_height=Inches(0.42),
              font_size=16, aligns="LLRRL")
    add_bullets(slide, [
        "Domain shift: CLIP learned from web photos. Satellite tiles and texture crops look "
        "nothing like them, so image features land far from the text features.",
        "Caption granularity: web captions say “a plane” or “a flower”, rarely the exact "
        "variant or species. The prompt names a class CLIP has barely seen named, so "
        "neighbouring classes get almost the same score.",
    ], top=Inches(4.45), spacing=8)
    set_notes(slide, f"""
Why is CLIP alone not enough? There are two different reasons, and they matter later.

The first is domain shift. CLIP was trained on web photos; satellite tiles in EuroSAT and
texture crops in DTD look nothing like them. CLIP gets {r.num(r.s('eurosat', 'clipzs', 'accuracy'))} percent on EuroSAT,
which has only {r.num(r.c('eurosat', 'n_classes'), 0)} classes.

The second is granularity. Web captions rarely name the exact aircraft variant or flower
species, so the text features for neighbouring fine-grained classes are close, and CLIP is
often torn between them. Aircraft, with {r.num(r.c('aircraft', 'n_classes'), 0)} variants, is at {r.num(r.s('aircraft', 'clipzs', 'accuracy'))} percent. Pets is the
counter-example: breed names are common online, and CLIP reaches {r.num(r.s('pets', 'clipzs', 'accuracy'))}.

Keep the second reason in mind: "torn between two neighbouring classes" is exactly the
situation our gate later targets.
""")
    return slide


def slide_landscape(prs, r):
    slide = new_slide(prs, "The method landscape: where TDA sits", kicker="1 · Background")
    rows = [
        ["CLIP zero-shot", "ICML'21", "none", "—", "no", "no"],
        ["Tip-Adapter", "ECCV'22", "few labeled shots", "—", "no", "no"],
        ["TPT", "NeurIPS'22", "none", "episodic", "yes", "no"],
        ["C-TPT", "ICLR'24", "none", "episodic", "yes", "yes"],
        ["O-TPT", "CVPR'25", "none", "episodic", "yes", "yes"],
        ["TDA", "CVPR'24", "none", "online", "no", "no"],
        ["DMN (weak)", "CVPR'24", "none", "online", "no", "no"],
        ["BoostAdapter", "NeurIPS'24", "none", "online", "no", "no"],
        ["ECALP", "ICLR'25", "none", "online", "no", "no"],
        ["UL-TTA", "arXiv'25", "none", "online", "no", "yes (temperature)"],
    ]
    add_table(slide, ["Method", "Venue", "Labels", "Adaptation", "Backprop", "Calibration-aware"], rows,
              MARGIN, Inches(1.62), Inches(8.3), col_widths=[1.6, 1.2, 1.7, 1.2, 1.0, 1.7],
              row_height=Inches(0.39), highlight_rows=(5,), align=PP_ALIGN.LEFT)
    add_text(slide, Inches(9.2), Inches(1.65), Inches(3.55), Inches(4.4), [
        ("Episodic", SUB, True, ACCENT),
        ("adapts on augmented views of each image with gradient steps, then resets.", SUB, False, INK),
        ("Online", SUB, True, ACCENT),
        ("keeps state (a cache, a memory) across the stream; forward passes only.", SUB, False, INK),
        ("Gap", SUB, True, WARN),
        ("calibration-aware methods so far are episodic and need backprop; the online, "
         "backprop-free family ignores calibration.", SUB, False, INK),
    ], spacing=6)
    add_text(slide, MARGIN, Inches(6.05), CONTENT_W, Inches(0.7), [
        ("Also relevant: D-TPT and TCA (episodic, calibration-aware, arXiv); DPE, FreeTTA, BCA "
         "(online); CRG (cache noise); ZS-Norm (logit-range calibration of adapters). "
         "Source: pm/research/related-work.md.", SMALL, False, MUTED)])
    set_notes(slide, """
Here is the landscape, so you can see why we picked TDA.

Tip-Adapter is the origin of the cache idea, but it needs a few labeled images per class,
so it is a different setting; we will use it later only as a reference for what labels buy.

The prompt-tuning family, TPT and its calibration-aware descendants C-TPT and O-TPT, is
episodic: for every single image it builds many augmented views, takes gradient steps on
the prompt, predicts, and resets. That needs backprop.

The online family, TDA, DMN, BoostAdapter and ECALP, never backpropagates and keeps a
memory across the stream. None of them is designed for calibration. UL-TTA is a recent
online method that does adjust a temperature without labels; it is the closest prior art
to our fourth change and we cite it.

So the gap is: calibration-aware methods exist, but only in the expensive episodic family.
""")
    return slide


def slide_why_tda(prs, r):
    slide = new_slide(prs, "Why we build on TDA", kicker="1 · Background")
    add_bullets(slide, [
        "It plays by our rules: online, no backprop, one view per image. It runs from cached "
        "image features on a CPU; the episodic methods need many views and gradient steps per image.",
        "It is the standard online baseline: CVPR'24, official code, and benchmarked for "
        "accuracy and calibration by the Illusion of Progress paper, whose codebase we fork.",
        "Its mechanism is explicit: a small cache of past test features. We can inspect "
        "exactly which stored image moved which prediction, and by how much.",
        "Our CLIP and TDA runs reproduce the benchmark's published ViT-B/16 accuracies "
        "(our baseline audit), so differences we report are ours, not a broken baseline.",
        "And it has a documented calibration problem, which the next sections dissect.",
    ], spacing=12)
    add_callout(slide, "TDA is the cleanest place to study how online TTA trades calibration for accuracy.")
    set_notes(slide, """
Why TDA, and not one of the others?

First, it fits the setting exactly: online, no gradients, one view per image. That also
makes it cheap: since CLIP is frozen and there is a single view, image features can be
computed once and replayed, which is how we will run all further experiments on a CPU.

Second, it is the standard baseline of the online family, with official code, and the
benchmark we fork already measured both its accuracy and its calibration.

Third, its mechanism is a cache we can open and inspect, so every claim about why it
becomes overconfident can be checked against what was actually stored.

We verified that our runs reproduce the published accuracies, so the baseline is sound.
""")
    return slide


# --- TDA section -------------------------------------------------------------------------
def slide_tda_idea(prs, r):
    slide = new_slide(prs, "TDA's idea: describe each class with test images, not only text",
                      kicker="2 · TDA in depth")
    panels = [(MARGIN, "Zero-shot CLIP", "one text feature per class"),
              (MARGIN + Inches(6.2), "TDA", "+ a few confident test features per class")]
    pw, ph, ptop = Inches(5.9), Inches(3.35), Inches(1.65)
    clusters = [  # (cx, cy) of the image cluster, (tx, ty) of the text anchor, colour
        ((1.7, 1.85), (0.85, 0.75), CLIPBLUE),
        ((4.15, 1.75), (4.85, 0.75), BASELINE),
    ]
    import random
    rng = random.Random(0)   # fixed layout of the decorative dots
    offsets = [(rng.uniform(-0.7, 0.7), rng.uniform(-0.55, 0.55)) for _ in range(9)]
    for index, (left, heading, sub) in enumerate(panels):
        add_panel(slide, left, ptop, pw, ph)
        add_text(slide, left + Inches(0.2), ptop + Inches(0.1), pw - Inches(0.4), Inches(0.4),
                 [[(heading, SUB, True, ACCENT if index else CLIPBLUE), ("   " + sub, SUB, False, MUTED)]])
        for (cx, cy), (tx, ty), color in clusters:
            for k, (dx, dy) in enumerate(offsets):
                cached = index == 1 and k < 3
                dot(slide, left + Inches(cx + dx), ptop + Inches(0.55 + cy + dy), Inches(0.24 if cached else 0.17),
                    color if not cached else SURFACE, outline=color if cached else None)
            dot(slide, left + Inches(tx), ptop + Inches(0.55 + ty), Inches(0.36), color,
                shape=MSO_SHAPE.STAR_5_POINT)
    add_text(slide, MARGIN, ptop + ph + Inches(0.1), CONTENT_W, Inches(0.4), [[
        ("★ text feature of a class     ", SUB, False, INK), ("• test images     ", SUB, False, INK),
        ("○ cached test images (pseudo-labeled)", SUB, False, INK)]])
    add_bullets(slide, [
        "Under domain shift the test images of a class sit away from its text feature. A few "
        "cached test features describe where this domain's images of the class actually are.",
        "A new image is scored against both: CLIP's text features and the cached image features.",
    ], top=Inches(5.55), spacing=4)
    set_notes(slide, """
Before the math, the idea in one picture. In zero-shot CLIP, each class is represented by
one point, its text feature, the star. Under domain shift the actual test images of the
class form a cloud somewhere else, and one point is a poor description of that cloud.

TDA's idea is to remember a few test images per class, the circled dots, which by
construction lie where this domain's images really are. A new image is then compared both
with the text features and with these remembered images. It is a nearest-neighbour vote
added on top of CLIP.

The catch, which we come back to after the full walkthrough, is that the remembered images
are labeled by CLIP's own guess.
""")
    return slide


def slide_tda_pipeline(prs, r):
    slide = new_slide(prs, "TDA, one image at a time", kicker="2 · TDA in depth")
    cap = r.num(r.c("dtd", "pos_capacity"), 0)
    steps = [
        ("1 · Encode", "CLIP logits z, softmax p, pseudo-label ŷ, entropy H"),
        ("2 · Write", "offer x to the positive cache under ŷ; mid-entropy x also to the negative one"),
        ("3 · Read", "similarity-weighted votes from both caches"),
        ("4 · Predict", "z + α·(positive votes) − α_neg·(negative votes)"),
    ]
    top, height = Inches(1.7), Inches(2.1)
    gap = Inches(0.32)
    width = (CONTENT_W - gap * 3) / 4
    left = MARGIN
    for index, (title, sub) in enumerate(steps):
        flow_box(slide, left, top, width, height, title, sub, fill=TINT_ACCENT if index in (1, 2) else SURFACE,
                 border=ACCENT if index in (1, 2) else RULE)
        if index < 3:
            arrow(slide, left + width + Inches(0.04), top + height / 2 - Inches(0.14), gap - Inches(0.08), Inches(0.28))
        left += width + gap
    add_bullets(slide, [
        f"Every image is offered to the positive cache; per class the {cap} lowest-entropy images are kept. "
        "There is no entropy threshold.",
        "Write happens before read: an admitted image is already in the cache when it is classified.",
        "Pseudo-labels and entropies come from raw CLIP. TDA's own output is never written back, "
        "so there is no feedback loop through the cache labels.",
        "State carries across the stream; nothing is ever reset or retrained.",
    ], top=Inches(4.1), spacing=10)
    set_notes(slide, f"""
Here is the loop TDA runs for every test image. First CLIP encodes the image and gives
logits, a distribution, a pseudo-label which is just CLIP's argmax, and an entropy.

Second, the write step: the image is offered to the positive cache under its pseudo-label,
and, if its entropy is in a middle band, also to a negative cache.

Third, the read step: both caches vote, weighted by similarity. Fourth, the prediction adds
the positive votes and subtracts the negative ones.

Three facts we verified in the code, because they are often stated wrongly. Every image is
offered; the {cap} most confident per class are kept. Writing happens before reading, so an image
can vote for itself. And the cache labels always come from raw CLIP, never from TDA's
output, so there is no feedback loop of TDA reinforcing its own decisions.
""")
    return slide


def slide_tda_write(prs, r):
    slide = new_slide(prs, "Write rule: what enters the positive cache", kicker="2 · TDA in depth")
    cap = r.num(r.c("dtd", "pos_capacity"), 0)
    add_panel(slide, MARGIN, Inches(1.65), CONTENT_W, Inches(1.55))
    add_equation(slide, r"\hat{y} = \arg\max_k z_k, \qquad H(x) = -\sum_k p_k \log p_k",
                 MARGIN, Inches(1.75), size=24, center_width=CONTENT_W)
    add_equation(slide, r"\mathcal{C}_{\hat{y}} \leftarrow \text{the } K \text{ lowest-}H"
                 r"\text{ items of } \mathcal{C}_{\hat{y}} \cup \{(f(x), H(x))\}",
                 MARGIN, Inches(2.45), size=24, center_width=CONTENT_W)
    add_bullets(slide, [
        f"K = {cap} per class. The key ŷ is CLIP's own guess, a pseudo-label: nothing checks it.",
        "When the class is full, x replaces the least confident stored item only if x has lower entropy.",
        "Early in the stream almost every image gets in, because classes are empty. Later an image must "
        "beat its class's current least confident item.",
        "Because write precedes read, an admitted x meets itself: cosine 1, vote exp(0) = 1, so its own "
        "pseudo-label gets +α before anything else is counted.",
    ], top=Inches(3.45), spacing=10)
    add_text(slide, MARGIN, Inches(6.45), CONTENT_W, Inches(0.35), [
        ("Code: online_method/tda.py, update_cache and adaptation_process.", SMALL, False, MUTED)])
    set_notes(slide, f"""
Now the write rule precisely. The pseudo-label is CLIP's argmax and the entropy is the
usual Shannon entropy of CLIP's distribution. The cache for the predicted class keeps the
{cap} lowest-entropy items it has seen.

Two consequences matter. At the start of the stream, classes are empty, so almost anything
gets in, including confident mistakes. Later, an image has to be more confident than the
weakest item already stored for its class.

And because the write happens before the read, an admitted image finds itself in the cache
with cosine one. Its vote is exactly one, and its own pseudo-label gets plus alpha. That
is a self-vote; keep it in mind for the calibration section.
""")
    return slide


def slide_tda_read(prs, r):
    slide = new_slide(prs, "Read rule: cached images vote, weighted by similarity", kicker="2 · TDA in depth")
    cap = r.num(r.c("dtd", "pos_capacity"), 0)
    add_panel(slide, MARGIN, Inches(1.65), Inches(6.6), Inches(1.75))
    add_equation(slide, r"c_k(x) = \sum_{i \in \mathcal{C}_k} \exp\!\left(-\beta\,(1 - f(x)^{\top} f_i)\right)",
                 MARGIN, Inches(1.8), size=24, center_width=Inches(6.6))
    add_equation(slide, r"z^{\mathrm{TDA}}_k = z_k + \alpha\, c_k(x) - \alpha_{\mathrm{neg}}\, n_k(x)",
                 MARGIN, Inches(2.65), size=24, center_width=Inches(6.6))
    add_bullets(slide, [
        "Features are L2-normalized, so f(x)ᵀf_i is a cosine. Each vote lies in (0, 1]; "
        "it equals 1 only for an identical feature.",
        f"Hence 0 ≤ c_k ≤ {cap}. Classes with no cached item get no vote at all.",
        "β sets how fast a vote fades as the images become less alike (right).",
    ], top=Inches(3.65), width=Inches(6.6), spacing=10)
    path = render_vote_figure(r, ASSETS / "vote_weight.png")
    add_picture(slide, path, Inches(7.45), Inches(1.6), Inches(5.3), Inches(4.4))
    add_text(slide, Inches(7.45), Inches(6.05), Inches(5.3), Inches(0.4), [
        ("β per dataset as used in our runs (from the TDA run records).", SMALL, False, MUTED)],
        align=PP_ALIGN.CENTER)
    set_notes(slide, f"""
The read rule. For each class, sum over its cached items an exponential of the cosine
between the new image and the item. The exponent is minus beta times one minus the cosine,
so an identical item gives exactly one, and less similar items give less.

With at most {cap} items per class, a class's cache score is between zero and {cap}. Classes
that have nothing cached get nothing.

The final TDA logit is CLIP's logit, plus alpha times this score, minus a small negative
cache term I will show in two slides.

The plot shows the vote weight against cosine for each dataset's beta. A larger beta means
only very similar cached images count.
""")
    return slide


def slide_tda_alpha_beta(prs, r):
    slide = new_slide(prs, "What α and β mean, and where their values come from", kicker="2 · TDA in depth")
    rows = []
    for d in DATASET_ORDER:
        rows.append([DATASET_LABEL[d], r.num(r.c(d, "n_classes"), 0), r.num(r.c(d, "alpha"), 1),
                     r.num(r.c(d, "beta"), 1), r.num(r.c(d, "vote_half_cos")),
                     r.num(r.c(d, "overturn_log_odds"), 1)])
    add_table(slide, ["Dataset", "Classes", "α", "β", "½-vote cos", "α·K"], rows,
              MARGIN, Inches(1.65), Inches(7.4), col_widths=[1.4, 0.9, 0.7, 0.7, 1.5, 1.6], font_size=16,
              row_height=Inches(0.44))
    add_bullets(slide, [
        "β (sharpness): how local the vote is. A vote is worth ½ at cos = 1 − ln 2 / β.",
        "α (residual weight): logit units per full vote; α·K is the most the cache can shift a logit "
        "difference (K = capacity).",
    ], top=Inches(4.5), width=Inches(7.4), spacing=8)
    add_panel(slide, Inches(8.35), Inches(1.65), Inches(4.38), Inches(4.4), fill=TINT_WARN, line=BASELINE)
    add_text(slide, Inches(8.55), Inches(1.8), Inches(4.0), Inches(4.1), [
        ("Provenance", SUB, True, WARN),
        ("These per-dataset values come from TDA's official repository, not the paper. "
         "How they were chosen is not stated; they may encode test-set information.", SUB, False, INK),
        (f"The paper's single searched setting is α = {r.num(r.c('dtd', 'p2_alpha'), 0)}, "
         f"β = {r.num(r.c('dtd', 'p2_beta'), 0)} (supp. Table 6).", SUB, False, INK),
        ("We report both: P1 uses the repo values, P2 the single setting (Results).", SUB, True, INK),
    ], spacing=8)
    set_notes(slide, f"""
What do alpha and beta mean? Beta is the sharpness: how local the nearest-neighbour vote is.
A vote halves when the cosine drops to one minus log two over beta, the column in the
middle. On EuroSAT, with beta {r.num(r.c('eurosat', 'beta'), 1)}, only images with cosine above {r.num(r.c('eurosat', 'vote_half_cos'))} count
much; on Aircraft, with beta {r.num(r.c('aircraft', 'beta'), 1)}, the vote is much broader.

Alpha is the weight of the cache against CLIP. The last column is the largest shift the
positive cache can apply to a logit difference, alpha times the capacity. On EuroSAT that
is {r.num(r.c('eurosat', 'overturn_log_odds'), 1)} logit units, which is a lot.

Important for fairness: these values are per dataset and come from TDA's repository, with
no stated procedure. The paper itself searched one setting, alpha {r.num(r.c('dtd', 'p2_alpha'), 0)}, beta {r.num(r.c('dtd', 'p2_beta'), 0)}. We
will report results under both choices.
""")
    return slide


def slide_tda_negative(prs, r):
    slide = new_slide(prs, "Negative cache: penalize classes that confused similar images",
                      kicker="2 · TDA in depth")
    lo, hi = r.num(r.c("dtd", "band_lower"), 1), r.num(r.c("dtd", "band_upper"), 1)
    pl = r.num(r.c("dtd", "mask_lower"))
    a_neg = r.num(r.c("dtd", "neg_alpha"), 3)
    b_neg = r.num(r.c("dtd", "neg_beta"), 1)
    ncap = r.num(r.c("dtd", "neg_capacity"), 0)
    add_panel(slide, MARGIN, Inches(1.65), CONTENT_W, Inches(1.6))
    add_equation(slide, r"\mathrm{write\ if}\quad \tau_l < H(x)/\log_2 C < \tau_h"
                 r"\qquad\mathrm{store}\ (f(x), H(x), p(x))",
                 MARGIN, Inches(1.75), size=24, center_width=CONTENT_W)
    add_equation(slide, r"n_k(x) = \sum_{j \in \mathcal{N}} \exp\!\left(-\beta_{\mathrm{neg}}(1 - f(x)^{\top} f_j)\right)"
                 r"\;\mathbf{1}\!\left[p_{j,k} > p_l\right]",
                 MARGIN, Inches(2.45), size=24, center_width=CONTENT_W)
    add_bullets(slide, [
        f"Written only for mid-entropy images: band τ_l = {lo}, τ_h = {hi}; {ncap} per pseudo-label; "
        "stores the whole distribution p.",
        f"Read: if x resembles an uncertain stored image, every class that image found plausible "
        f"(p > {pl}) loses α_neg·vote, with α_neg = {a_neg}, β_neg = {b_neg}, fixed on all datasets.",
        "Effect: a small damping among confusable classes. It is a correction on CLIP's side; "
        "none of our changes touch it.",
        "The paper's equations contain a double negative; the code masks p > p_l and subtracts. "
        "We follow the code.",
    ], top=Inches(3.5), spacing=8)
    set_notes(slide, f"""
TDA has a second cache. It stores images whose normalized entropy falls in a middle band,
between {lo} and {hi}: uncertain, but not hopeless. It stores the whole distribution of each.

At read time, if the new image resembles one of these uncertain images, every class that
the stored image found plausible, with probability above {pl}, gets a penalty of {a_neg} times the
vote. It is a small damping term among confusable classes.

Two notes. The paper's equation has a double negative; we follow the code, which masks
probabilities above the threshold and subtracts. And none of our changes touch this cache:
it stays on CLIP's side in every variant.
""")
    return slide


def slide_tda_poe(prs, r):
    slide = new_slide(prs, "Final logits are a normalized product of experts", kicker="2 · TDA in depth")
    add_panel(slide, MARGIN, Inches(1.65), CONTENT_W, Inches(1.75))
    add_equation(slide, r"\mathrm{softmax}(z + \alpha c)_k \;=\; \frac{p_k\; q_k^{\,\alpha}}{\sum_j\, p_j\; q_j^{\,\alpha}},"
                 r"\qquad p = \mathrm{softmax}(z),\quad q = \mathrm{softmax}(c)",
                 MARGIN, Inches(1.75), size=30, center_width=CONTENT_W)
    add_text(slide, MARGIN, Inches(2.85), CONTENT_W, Inches(0.4), [
        ("because e^(z_k + α c_k) = e^(z_k) · (e^(c_k))^α and the normalizers of p and q cancel; "
         "z already includes the negative-cache term", SMALL, False, MUTED)], align=PP_ALIGN.CENTER)
    alpha = float(r.c("dtd", "p2_alpha"))
    p, q = ILLUSTRATIVE["toy_p"], ILLUSTRATIVE["toy_q_agree"]
    prod = product_of_experts(p, q, alpha)
    mix = mixture(p, q)
    rows = [[f"class {i + 1}", r.illus(p[i]), r.illus(q[i]), r.illus(prod[i]), r.illus(mix[i])] for i in range(3)]
    add_table(slide, ["Illustrative", "CLIP p", "cache q", f"TDA  p·q^α  (α = {r.num(alpha, 0)})", "50/50 mixture"],
              rows, MARGIN, Inches(3.65), Inches(7.2), col_widths=[1.3, 1.0, 1.0, 1.9, 1.4], font_size=16,
              highlight_rows=(0,), row_height=Inches(0.44))
    add_bullets(slide, [
        "Experts multiply: where CLIP and the cache agree, the product is sharper than either.",
        f"Both say {r.illus(p[0])}; TDA says {r.illus(prod[0])}.",
        "The cache is built from CLIP's own guesses, so agreement is the common case.",
    ], left=Inches(8.15), top=Inches(3.6), width=Inches(4.6), spacing=8)
    set_notes(slide, f"""
This slide is the key to the whole talk. Adding alpha times the cache score to the logits
and then taking a softmax is exactly the same as multiplying CLIP's distribution by the
cache's distribution raised to the power alpha, and renormalizing. The two softmax
normalizers simply cancel. That is called a normalized product of experts.

Products of experts have a well-known property: when the experts agree, the result is
sharper than either. In this illustrative example, CLIP and the cache both give the top
class {r.illus(p[0])}. TDA, with alpha {r.num(alpha, 0)}, reports {r.illus(prod[0])}. A simple average of the two would stay at
{r.illus(mix[0])}.

And because the cache is filled with CLIP's own guesses, agreement is the normal case, also
when CLIP is wrong. Remember the right-hand column; it is our second change.
""")
    return slide


def _worked_example(r):
    """The worked example's numbers: Pets' alpha/beta from data, illustrative cosines."""
    scale = r.litv("clip_logit_scale")
    alpha, beta = float(r.c("pets", "alpha")), float(r.c("pets", "beta"))
    z = [scale * c for c in ILLUSTRATIVE["we_text_cos"]]
    p = softmax(z)
    votes_other = [sum(math.exp(-beta * (1 - c)) for c in cs) for cs in ILLUSTRATIVE["we_cache_cos"]]
    pred = max(range(3), key=lambda k: z[k])
    votes = list(votes_other)
    votes[pred] += 1.0          # the self-vote: x was written before it is read
    z_tda = [zk + alpha * ck for zk, ck in zip(z, votes)]
    z_noself = [zk + alpha * ck for zk, ck in zip(z, votes_other)]
    return dict(scale=scale, alpha=alpha, beta=beta, z=z, p=p, votes=votes, z_tda=z_tda,
                p_tda=softmax(z_tda), p_noself=softmax(z_noself), pred=pred)


def slide_tda_example(prs, r):
    slide = new_slide(prs, "Worked example: one Pets image through TDA (illustrative)", kicker="2 · TDA in depth")
    w = _worked_example(r)
    names = ILLUSTRATIVE["we_classes"]
    rows = []
    for k in range(3):
        cos_items = ILLUSTRATIVE["we_cache_cos"][k]
        cached = ", ".join(r.illus(c) for c in cos_items) if cos_items else "—"
        if k == w["pred"]:
            cached = (cached + " + itself (cos " + r.illus(1.0) + ")") if cos_items else "itself"
        rows.append([names[k], r.illus(w["z"][k], 1), r.illus(w["p"][k]), cached, r.illus(w["votes"][k]),
                     r.illus(w["z_tda"][k], 1), r.illus(w["p_tda"][k])])
    add_table(slide, ["Class", "CLIP z", "CLIP p", "cached items (cos to x)", "votes c", "z + α c", "TDA p"],
              rows, MARGIN, Inches(1.65), CONTENT_W, col_widths=[1.2, 0.9, 0.9, 3.0, 1.0, 1.1, 1.0],
              font_size=16, row_height=Inches(0.46), highlight_rows=(w["pred"],))
    add_bullets(slide, [
        f"Illustrative cosines; α = {r.num(w['alpha'], 1)}, β = {r.num(w['beta'], 1)} are Pets' values "
        f"and s = {r.num(w['scale'], 0)}. Negative cache omitted.",
        f"CLIP: {r.illus(w['p'][w['pred']])} on {names[w['pred']]}. The image is written under that "
        f"pseudo-label first, so it votes for itself.",
        f"TDA: {r.illus(w['p_tda'][w['pred']])}; without the self-vote, {r.illus(w['p_noself'][w['pred']])}. "
        f"The self-vote is {r.illus(100 / w['votes'][w['pred']], 0)}% of {names[w['pred']]}'s cache score.",
        "Nothing here depends on whether Birman is the right answer.",
    ], top=Inches(3.75), spacing=8)
    set_notes(slide, f"""
Let me put numbers on it, with an illustrative Pets image. Three candidate breeds. CLIP's
cosines give logits about one point apart, and CLIP says {names[w['pred']]} with {r.illus(w['p'][w['pred']])}.

The cache already holds two {names[0]} images and one {names[1]}. Because the image is written before
it is read, it is also in the {names[w['pred']]} slot itself, with cosine one and vote one. With Pets'
alpha {r.num(w['alpha'], 1)} and beta {r.num(w['beta'], 1)}, the {names[0]} logit rises by alpha times its votes.

TDA's confidence becomes {r.illus(w['p_tda'][w['pred']])}. Without the self-vote it would have been {r.illus(w['p_noself'][w['pred']])}. In logit terms the
self-vote is {r.illus(100 / w['votes'][w['pred']], 0)} percent of the {names[w['pred']]} cache score: a large part of the boost is the image
agreeing with itself. And nothing in this computation knows whether
{names[w['pred']]} is right. That is the mechanism of overconfidence, in miniature.
""")
    return slide


def slide_tda_gains(prs, r):
    slide = new_slide(prs, "What TDA buys: accuracy over zero-shot CLIP", kicker="2 · TDA in depth")
    rows = []
    for d in DATASET_ORDER:
        clip, tda = r.s(d, "clipzs", "accuracy"), r.s(d, "tda", "accuracy")
        rows.append([DATASET_LABEL[d], r.num(r.c(d, "n_classes"), 0), r.num(r.c(d, "n_samples"), comma=True),
                     r.num(clip), r.num(tda), r.num(tda - clip, sign=True)])
    clip_m, tda_m = r.s_mean("clipzs", "accuracy"), r.s_mean("tda", "accuracy")
    rows.append(["Mean", "", r.num(r.total_images(), comma=True), r.num(clip_m), r.num(tda_m),
                 r.num(tda_m - clip_m, sign=True)])
    add_table(slide, ["Dataset", "Classes", "Test images", "CLIP top-1 (%)", "TDA top-1 (%)", "Gain"], rows,
              MARGIN, Inches(1.65), Inches(8.0), col_widths=[1.5, 1.0, 1.3, 1.4, 1.4, 1.0], font_size=16,
              row_height=Inches(0.45), highlight_rows=(5,))
    gains = {d: r.s(d, "tda", "accuracy") - r.s(d, "clipzs", "accuracy") for d in DATASET_ORDER}
    best, worst = max(gains, key=gains.get), min(gains, key=gains.get)
    n_pos = sum(g > 0 for g in gains.values())
    add_bullets(slide, [
        f"TDA gains on {r.num(n_pos, 0)} of {r.num(len(gains), 0)} datasets.",
        f"Most on {DATASET_LABEL[best]} ({r.num(gains[best], sign=True)}): {r.num(r.c(best, 'n_classes'), 0)} classes "
        f"and the largest α.",
        f"Least on {DATASET_LABEL[worst]} ({r.num(gains[worst], sign=True)}): CLIP is right only "
        f"{r.num(r.s(worst, 'clipzs', 'accuracy'), 1)}% of the time, so most pseudo-labels are wrong.",
    ], left=Inches(8.95), top=Inches(1.7), width=Inches(3.8), spacing=12)
    add_text(slide, MARGIN, Inches(5.0), Inches(8.0), Inches(0.8), [
        ("P1 hyperparameters (TDA's per-dataset α/β), ViT-B/16, seed 0, CoOp test splits. "
         "Source: results/summary.csv.", SMALL, False, MUTED)])
    add_callout(slide, "TDA's accuracy gain is real but uneven. Next: what it costs in calibration.",
                top=Inches(6.05))
    set_notes(slide, f"""
Before any criticism, what TDA achieves. On all five datasets TDA is more accurate than
zero-shot CLIP, on average from {r.num(clip_m)} to {r.num(tda_m)} percent.

The gain is very uneven. On {DATASET_LABEL[best]} it is {r.num(gains[best])} points; there are only {r.num(r.c(best, 'n_classes'), 0)} classes, so each
class's cache fills quickly, and the repository's alpha is the largest. On {DATASET_LABEL[worst]} it is only
{r.num(gains[worst])} points: CLIP is right only about a quarter of the time there, so most cached
pseudo-labels are wrong.

These are protocol P1 numbers, with TDA's per-dataset hyperparameters, single seed.
""")
    return slide


def slide_tda_gain_source(prs, r):
    slide = new_slide(prs, "Where TDA's gain comes from: overturning CLIP's top choice", kicker="2 · TDA in depth")
    add_panel(slide, MARGIN, Inches(1.65), Inches(6.5), Inches(1.3))
    add_equation(slide, r"\mathrm{class\ 2\ overtakes\ class\ 1}\ \Leftrightarrow\ \log\frac{p_1}{p_2} < \alpha\,(c_2 - c_1)\ \leq\ \alpha K",
                 MARGIN, Inches(1.8), size=26, center_width=Inches(6.5))
    rows = [[DATASET_LABEL[d], r.num(r.c(d, "alpha"), 1), r.num(r.c(d, "overturn_log_odds"), 1),
             r.num(r.c(d, "overturn_odds"), comma=True) + " : 1"] for d in DATASET_ORDER]
    add_table(slide, ["Dataset", "α", "α·K", "max CLIP odds overturned"], rows, MARGIN, Inches(3.1),
              Inches(6.5), col_widths=[1.4, 0.7, 0.8, 2.6], font_size=16, row_height=Inches(0.4))
    add_text(slide, MARGIN, Inches(5.6), Inches(6.5), Inches(0.5), [
        ("p from CLIP (with negative cache); K = capacity. Accuracy changes only where the argmax flips.",
         SMALL, False, MUTED)])
    add_placeholder(slide, Inches(7.45), Inches(1.65), Inches(5.28), Inches(4.25), "A2",
                    "argmax flips vs CLIP's top-2 gap",
                    "Per dataset: how many predictions TDA flips, how many wrong→right vs right→wrong, "
                    "histogrammed by CLIP's top-2 probability gap. Tests whether the gain lives on near-ties.")
    add_callout(slide, "Hypothesis to test: TDA's gain is concentrated where CLIP is torn between two classes.",
                top=Inches(6.15))
    set_notes(slide, f"""
Accuracy changes only when TDA flips CLIP's argmax. A flip from class one to class two
happens exactly when CLIP's log-odds between them is smaller than alpha times the cache's
preference for class two. Since each class's cache score is at most the capacity, there is
a hard limit on how confident a CLIP decision can be overturned.

The table shows that limit. With alpha {r.num(r.c('dtd', 'alpha'), 1)}, the cache can overturn CLIP decisions up to odds of
about {r.num(r.c('dtd', 'overturn_odds'), comma=True)} to one. On EuroSAT, with alpha {r.num(r.c('eurosat', 'alpha'), 1)}, essentially any decision can be overturned.

What we do not know yet is where the flips actually happen. Evidence A2 will histogram
flips by CLIP's top-two gap. Our hypothesis, which the gate relies on, is that the useful
flips are concentrated on near-ties.
""")
    return slide


# --- Calibration section ---------------------------------------------------------------
def slide_ece(prs, r):
    slide = new_slide(prs, "Confidence, calibration and ECE", kicker="3 · Calibration")
    bins = r.num(r.n_bins(), 0)
    add_panel(slide, MARGIN, Inches(1.65), Inches(7.5), Inches(1.95))
    add_equation(slide, r"\mathrm{ECE} = \sum_{b=1}^{B} \frac{n_b}{N}\,\left|\,\mathrm{acc}(b) - \mathrm{conf}(b)\,\right|",
                 MARGIN, Inches(1.7), size=24, center_width=Inches(7.5))
    add_equation(slide, r"\mathrm{gap} = \overline{\mathrm{conf}} - \mathrm{acc},\qquad \mathrm{ECE} \geq |\mathrm{gap}|",
                 MARGIN, Inches(2.9), size=24, center_width=Inches(7.5))
    add_bullets(slide, [
        "Confidence = probability of the predicted class (top-label).",
        "Calibrated: among predictions made with confidence c, a fraction c is correct.",
        f"ECE: sort predictions into B = {bins} equal-width confidence bins; average |accuracy − mean "
        "confidence| per bin, weighted by bin size.",
        "Gap > 0: overconfident; gap < 0: underconfident. ECE ≈ |gap| means every bin errs the "
        "same way: a uniform shift.",
    ], top=Inches(3.8), width=Inches(7.5), spacing=6)
    path = render_calibration_figure(ASSETS / "what_is_calibration.png")
    add_picture(slide, path, Inches(8.4), Inches(1.6), Inches(4.35), Inches(4.2))
    add_text(slide, Inches(8.4), Inches(5.85), Inches(4.35), Inches(0.4), [
        ("Illustrative reliability diagram (synthetic curves).", SMALL, False, MUTED)], align=PP_ALIGN.CENTER)
    set_notes(slide, f"""
Now calibration. The confidence of a prediction is the probability of the predicted class.
A model is calibrated if, among all predictions it makes with, say, seventy percent
confidence, seventy percent are correct.

Expected calibration error measures the violation. We sort predictions into {bins} equal-width
bins by confidence, take the absolute difference between accuracy and mean confidence in
each bin, and average with bin sizes as weights. We use top-label ECE throughout.

We will also use a simpler number, the gap: mean confidence minus accuracy. By the triangle
inequality ECE is at least the absolute gap. When the two are almost equal, every bin errs
in the same direction, which means the miscalibration is a uniform shift. That observation
will motivate our temperature change.

The diagram on the right is a synthetic reliability diagram: the diagonal is perfect
calibration, a curve below it is overconfident, and ECE averages the vertical gaps.
""")
    return slide


def slide_why_calibration(prs, r):
    slide = new_slide(prs, "Why calibration matters", kicker="3 · Calibration")
    fg_c, fg_t = r.num(r.litv("illusion_rn50_finegrained_clip_ece")), r.num(r.litv("illusion_rn50_finegrained_tda_ece"))
    all_c, all_t = r.num(r.litv("illusion_rn50_all_clip_ece")), r.num(r.litv("illusion_rn50_all_tda_ece"))
    add_bullets(slide, [
        "Decisions use the confidence: accept automatically above a threshold, abstain or ask a human "
        "below it.",
        (1, "Overconfident: wrong answers pass the filter. Underconfident: right answers are needlessly deferred."),
        "In online TTA the confidence also steers the method itself: TDA's cache keeps its most "
        "confident images.",
        "Accuracy alone hides this: two models with equal accuracy can be very differently trustworthy.",
    ], width=Inches(7.0), spacing=10)
    add_panel(slide, Inches(7.95), Inches(1.65), Inches(4.78), Inches(4.3), fill=TINT_WARN, line=BASELINE)
    add_text(slide, Inches(8.15), Inches(1.8), Inches(4.4), Inches(4.05), [
        ("The Illusion of Progress? (NeurIPS'25)", SUB, True, WARN),
        ("A benchmark of TTA for vision-language models: accuracy gains often come with worse "
         "calibration.", SUB, False, INK),
        ([("TDA, ResNet-50, fine-grained average ECE: ", SUB, False, INK),
          (f"{fg_c} → {fg_t}", SUB, True, WARN)]),
        ([("All datasets: ", SUB, False, INK), (f"{all_c} → {all_t}", SUB, True, WARN),
          ("  (CLIP → TDA, Table 7)", SUB, False, INK)]),
        ("ECE is reported only for ResNet-50; our ViT-B/16 ECE has no published counterpart.",
         SUB, False, MUTED),
    ], spacing=8)
    set_notes(slide, f"""
Why should anyone care about calibration? Because the confidence is what downstream
decisions use. A typical system accepts confident predictions automatically and sends the
rest to a human. If the model is overconfident, wrong answers slip through; if it is
underconfident, good answers are wasted on human review.

In online test-time adaptation there is a second reason: the confidence also steers the
method. TDA's cache keeps the most confident images, so miscalibration changes what it
remembers.

The benchmark we build on measured exactly this. For TDA with a ResNet-50 backbone, the
average ECE over the fine-grained datasets goes from {fg_c} for CLIP to {fg_t}, and over all
datasets from {all_c} to {all_t}. They report ECE only for ResNet-50, so our ViT-B/16 numbers,
coming next, have no published counterpart.
""")
    return slide


def slide_clip_vs_tda_cal(prs, r):
    slide = new_slide(prs, "CLIP vs TDA: accuracy, confidence and ECE per dataset", kicker="3 · Calibration")
    rows = []
    for d in DATASET_ORDER:
        rows.append([DATASET_LABEL[d]] + [r.num(r.r("headline", d, m, col)) for m in ("clipzs", "tda")
                                          for col in ("accuracy", "mean_confidence", "ece")]
                    + [r.num(r.r("headline", d, "tda", "gap"), sign=True)])
    rows.append(["Mean"] + [r.num(r.r_mean("headline", m, col)) for m in ("clipzs", "tda")
                            for col in ("accuracy", "mean_confidence", "ece")]
                + [r.num(r.r_mean("headline", "tda", "gap"), sign=True)])
    add_table(slide, ["Dataset", "CLIP acc", "CLIP conf", "CLIP ECE", "TDA acc", "TDA conf", "TDA ECE", "TDA gap"],
              rows, MARGIN, Inches(1.65), CONTENT_W, col_widths=[1.5, 1, 1, 1, 1, 1, 1, 1], font_size=16,
              row_height=Inches(0.42), highlight_rows=(5,))
    over = [d for d in DATASET_ORDER if r.r("headline", d, "tda", "gap") > 0]
    under = [d for d in DATASET_ORDER if d not in over]
    rise = r.s_mean("tda", "ece") - r.s_mean("clipzs", "ece")
    add_bullets(slide, [
        f"TDA is overconfident on {r.num(len(over), 0)} of {r.num(len(DATASET_ORDER), 0)} datasets, and there its ECE is "
        "almost exactly its gap: a near-uniform upward shift of confidence.",
        f"{', '.join(DATASET_LABEL[d] for d in under)}: CLIP is underconfident "
        f"(gap {r.num(r.r('headline', under[0], 'clipzs', 'gap'), sign=True)}), and TDA's shift happens to land "
        "near calibration.",
    ], top=Inches(5.2), spacing=6)
    add_text(slide, MARGIN, Inches(4.75), CONTENT_W, Inches(0.35), [
        (f"All in %. conf = mean top-1 confidence; gap = conf − acc. Mean ECE rises by "
         f"{r.num(rise)} points. Source: results/analysis/reliability_stats.csv (evidence A1, seed 0).",
         SMALL, False, MUTED)])
    set_notes(slide, f"""
Here are our own numbers, which is evidence A1, computed from the per-bin data in the run
records. For each dataset: accuracy, mean confidence and ECE for CLIP and for TDA, and
TDA's gap.

TDA raises mean ECE from {r.num(r.s_mean('clipzs', 'ece'))} to {r.num(r.s_mean('tda', 'ece'))}. On {r.num(len(over), 0)} of the five datasets TDA is overconfident, and
look at the last two columns: the ECE is almost exactly the gap. On DTD, ECE {r.num(r.r('headline', 'dtd', 'tda', 'ece'))}, gap
{r.num(r.r('headline', 'dtd', 'tda', 'gap'))}. So TDA shifts confidence up nearly uniformly.

Pets is the exception. There CLIP itself is underconfident, and TDA's upward shift happens
to land close to calibration. Keep Pets in mind; it behaves differently in every change.
""")
    return slide


def slide_reliability(prs, r):
    slide = new_slide(prs, "Reliability diagrams: TDA moves the curve below the diagonal", kicker="3 · Calibration")
    w = Inches(2.38)
    gap = (CONTENT_W - 5 * w) / 4
    for index, d in enumerate(DATASET_ORDER):
        add_picture(slide, FIGURES / f"deck_reliability_{d}.png", MARGIN + index * (w + gap), Inches(1.7), w, w)
    add_text(slide, MARGIN, Inches(4.2), CONTENT_W, Inches(0.45), [[
        ("●  CLIP zero-shot", SUB, True, CLIPBLUE), ("        ■  TDA", SUB, True, BASELINE),
        ("        - - -  perfect calibration", SUB, False, MUTED)]], align=PP_ALIGN.CENTER)
    add_bullets(slide, [
        "Below the diagonal = accuracy lower than the confidence claimed.",
        "On DTD, EuroSAT and Aircraft, TDA's whole curve drops below CLIP's: confidence went up in "
        "every bin, accuracy did not follow.",
        "On Pets, CLIP sits above the diagonal (underconfident); TDA moves it onto the diagonal.",
    ], top=Inches(4.8), spacing=6)
    set_notes(slide, """
The same story as reliability diagrams, one per dataset. The dashed diagonal is perfect
calibration. Blue is CLIP, orange is TDA. Points below the diagonal mean the model claims
more confidence than its accuracy supports.

On DTD, EuroSAT and Aircraft, TDA's curve is below CLIP's across the whole range: it pushes
confidence up everywhere, and accuracy does not keep pace. On Pets, CLIP's curve is above
the diagonal, and TDA's upward push brings it onto the line.

So TDA does not make a few wild mistakes; it inflates confidence across the board. The next
slide explains where that inflation comes from.
""")
    return slide


def slide_anatomy(prs, r):
    slide = new_slide(prs, "Anatomy of TDA's overconfidence", kicker="3 · Calibration")
    add_bullets(slide, [
        "Self-vote: write precedes read, so an admitted image adds +α to its own pseudo-label.",
        "Product of experts: the cache holds CLIP's own guesses, so it mostly agrees with CLIP, "
        "and agreement multiplies.",
        "Persistent wrong labels: a confidently wrong cached image votes for its wrong class on every "
        "similar later image.",
        "None depends on correctness; all raise confidence.",
    ], width=Inches(6.6), size=SUB, spacing=8)
    pairs = ", ".join(f"{DATASET_LABEL[d]} {r.num(r.r('headline', d, 'tda', 'ece'), 1)} vs "
                      f"{r.num(r.r('headline', d, 'tda', 'gap'), 1)}" for d in ("dtd", "eurosat", "aircraft"))
    add_text(slide, MARGIN, Inches(5.2), Inches(6.6), Inches(0.85), [
        ("Measured so far, TDA's ECE vs gap: " + pairs + ".", SUB, False, INK)])
    add_placeholder(slide, Inches(7.5), Inches(1.65), Inches(5.23), Inches(2.1), "A3",
                    "self-vote share and cache purity, feature replay",
                    "Over the stream: share of TDA's confidence lift due to self-votes; fraction of "
                    "cached items with a wrong pseudo-label; class coverage.")
    add_placeholder(slide, Inches(7.5), Inches(3.95), Inches(5.23), Inches(2.15), "A4",
                    "one poisoned cached item",
                    "A wrongly pseudo-labeled cached image, the later images it pulled, and CLIP's vs "
                    "the cache's logit contribution for each.")
    add_callout(slide, "Diagnosis: the cache adds confidence regardless of correctness. Each of our changes "
                "targets one of these links.", top=Inches(6.22))
    set_notes(slide, f"""
Now, after the full walkthrough, the weaknesses. Three mechanisms inflate TDA's confidence.

First, the self-vote we saw: an admitted image votes for its own pseudo-label before it is
classified.

Second, the product of experts: the cache is made of CLIP's own guesses, so it tends to
agree with CLIP, and agreement multiplies.

Third, a confidently wrong image stays in the cache under the wrong label and keeps voting
for it, until it is displaced by an even more confident image of the same predicted class.

None of these depends on whether the prediction is correct. What we already see in the
data is that TDA's ECE equals its gap, for example {pairs}.
What we will add, boxes A3 and A4, is the direct evidence: how much of the lift is
self-votes, how pure the cache is, and a concrete poisoned item with the images it pulled.
""")
    return slide


# --- Proposal -------------------------------------------------------------------------------
def slide_proposal(prs, r):
    slide = new_slide(prs, "Our proposal: four training-free changes, one link each", kicker="4 · Our proposal")
    tau = r.num(r.margin_cfg["margin_threshold"], 1)
    wgt = r.num(r.gate_cfg["fusion_weight"], 1)
    rows = [
        ["1  Margin admission", "what enters the cache", f"admit only if CLIP's top-2 logit margin > {tau}",
         "dropped (negative result)"],
        ["2  Probabilistic fusion", "how cache and CLIP combine",
         f"mixture (1 − w)·p + w·q, w = {wgt}, instead of the product", "fails alone: underconfident"],
        ["3  Gated fusion", "when the cache is consulted", "mixture only on two-way ties, CLIP elsewhere",
         "recommended"],
        ["4  LOO temperature", "the final scale", "T ≥ 1 matched to a label-free accuracy estimate",
         "helps alone, does not stack"],
    ]
    colors = {(0, 3): WARN, (1, 3): WARN, (2, 3): GOOD, (3, 3): BASELINE}
    add_table(slide, ["Change", "Targets", "Idea", "Outcome"], rows, MARGIN, Inches(1.7), CONTENT_W,
              col_widths=[2.2, 2.1, 4.4, 2.3], font_size=16, row_height=Inches(0.62), align=PP_ALIGN.LEFT,
              bold_first_col=True, cell_colors=colors)
    add_bullets(slide, [
        "Same constraints as TDA: no labels, no backprop, one pass; the encoders and the negative cache are untouched.",
        "Each change is measured alone against “none” = TDA run through our code with all four "
        "changes switched off, in the same sweep and seed.",
        "Every knob was fixed before the runs, except one choice among fusion variants (disclosed later).",
    ], top=Inches(4.95), spacing=6, size=SUB)
    set_notes(slide, f"""
Our proposal: four small changes, each aimed at one link of the diagnosis. Margin admission
changes what enters the cache. Probabilistic fusion changes how the cache and CLIP are
combined: a mixture instead of a product. Gated fusion changes when the cache is consulted
at all. And the leave-one-out temperature rescales the final confidence.

The right column gives away the ending: the first is a negative result, the second fails on
its own but leads to the third, which is our recommendation, and the fourth helps alone but
does not combine with the third.

Every change is measured on its own against the same baseline, which we call "none": TDA run
through our code with every change switched off, in the same sweep. Our knobs were fixed
before running, with one exception about the gate that I will disclose.
""")
    return slide


def slide_margin(prs, r):
    slide = new_slide(prs, "Change 1 · Margin admission: a negative result", kicker="4 · Our proposal")
    tau = r.num(r.margin_cfg["margin_threshold"], 1)
    rows = []
    for d in DATASET_ORDER:
        rows.append([DATASET_LABEL[d], r.num(r.admission[d], 1) + "%",
                     r.num(r.v(d, "none", "accuracy")), r.num(r.v(d, "margin", "accuracy")),
                     r.num(r.r("ablation", d, "none", "gap"), 1, sign=True),
                     r.num(r.r("ablation", d, "margin", "gap"), 1, sign=True),
                     r.num(r.v(d, "none", "ece")), r.num(r.v(d, "margin", "ece"))])
    rows.append(["Mean", r.num(sum(r.admission.values()) / len(r.admission), 1) + "%",
                 r.num(r.v_mean("none", "accuracy")), r.num(r.v_mean("margin", "accuracy")),
                 r.num(r.r_mean("ablation", "none", "gap"), 1, sign=True),
                 r.num(r.r_mean("ablation", "margin", "gap"), 1, sign=True),
                 r.num(r.v_mean("none", "ece")), r.num(r.v_mean("margin", "ece"))])
    add_text(slide, MARGIN, Inches(1.6), CONTENT_W, Inches(0.75), [
        [("Motive: ", SUB, True, ACCENT), ("TDA lets every image compete for the cache, confident mistakes included.  ",
                                           SUB, False, INK),
         ("Why it should work: ", SUB, True, ACCENT),
         (f"a large top-2 logit margin m = z(1) − z(2) > {tau} should keep out ambiguous images.", SUB, False, INK)]])
    add_table(slide, ["Dataset", "admitted", "acc: none", "margin", "gap: none", "margin", "ECE: none",
                      "margin"], rows, MARGIN, Inches(2.45), Inches(8.1),
              col_widths=[1.35, 1.05, 1.05, 0.9, 1.05, 0.9, 1.05, 0.9], font_size=14, row_height=Inches(0.36),
              highlight_rows=(5,))
    low = min(r.admission, key=r.admission.get)
    high = max(r.admission, key=r.admission.get)
    all_worse = all(r.v(d, "margin", "accuracy") < r.v(d, "none", "accuracy")
                    and r.v(d, "margin", "ece") > r.v(d, "none", "ece") for d in DATASET_ORDER)
    add_bullets(slide, [
        "Worse on every dataset, both metrics." if all_worse else "Worse on most datasets.",
        f"Admission follows CLIP's accuracy: {r.num(r.admission[low], 1)}% on {DATASET_LABEL[low]}, "
        f"{r.num(r.admission[high], 1)}% on {DATASET_LABEL[high]}.",
        "A starved cache is a biased one: only easy classes get in, and pull everything.",
        "Margin measures loudness, not correctness.",
    ], left=Inches(8.95), top=Inches(2.45), width=Inches(3.8), size=SUB, spacing=8)
    add_text(slide, MARGIN, Inches(5.05), Inches(8.1), Inches(0.6), [
        ("admitted = share of the stream that passed the margin gate and competed for a cache slot "
         "(TDA: every image).  none = TDA through our code, all changes off.", SMALL, False, MUTED)])
    add_placeholder(slide, MARGIN, Inches(5.75), Inches(8.1), Inches(1.0), "A5",
                    "cache purity and class coverage, with vs without margin",
                    "Wrongly pseudo-labeled cached images before/after the gate.")
    set_notes(slide, f"""
The first change, as one compact slide, because it is a negative result.

The motive: TDA lets every image compete for the cache, including confident mistakes. The
idea: require a top-two logit margin above {tau}, one global threshold fixed in advance, so that
ambiguous images stay out.

It is worse on every dataset, on both accuracy and ECE, on average ECE {r.num(r.v_mean('none', 'ece'))} to {r.num(r.v_mean('margin', 'ece'))}. Two
definitions for the table: "admitted" is the share of the stream that passed the gate and
competed for a cache slot, which is every image for TDA; "none" is TDA through our code.

Why it fails: the admission rate follows CLIP's accuracy, {r.num(r.admission[low], 1)} percent on {DATASET_LABEL[low]}, {r.num(r.admission[high], 1)} on
{DATASET_LABEL[high]}. When few images get in, they come from the classes CLIP already finds easy, and
the cache then pulls every prediction toward those classes: on {DATASET_LABEL[low]} the gap goes from
{r.num(r.r('ablation', low, 'none', 'gap'), 1)} to {r.num(r.r('ablation', low, 'margin', 'gap'), 1)} points. And fundamentally, a confidently wrong image has a large margin too:
the margin measures loudness, not correctness. Evidence A5 will show cache purity and
coverage directly. We dropped this change.
""")
    return slide


def slide_fusion_motive(prs, r):
    slide = new_slide(prs, "Change 2 · Probabilistic fusion: motive and why it should work", kicker="4 · Our proposal")
    wgt = r.num(r.gate_cfg["fusion_weight"], 1)
    alpha = float(r.c("dtd", "p2_alpha"))
    p = ILLUSTRATIVE["toy_p"]
    agree, disagree = ILLUSTRATIVE["toy_q_agree"], ILLUSTRATIVE["toy_q_disagree"]
    add_table(slide, ["", "TDA: normalized product", "Ours: mixture"], [
        ["rule", "p_k · q_k^α / Σ_j p_j q_j^α", f"(1 − w)·p_k + w·q_k,  w = {wgt}"],
        ["experts agree", "sharper than either", "between the two"],
        ["experts disagree", "the more peaked wins", "a compromise"],
        ["upper bound", "none: can reach 1", "max(max p, max q)"],
    ], MARGIN, Inches(1.7), Inches(7.4), col_widths=[1.6, 2.9, 2.9], font_size=16, row_height=Inches(0.5),
        align=PP_ALIGN.LEFT, bold_first_col=True)
    pa, ma = product_of_experts(p, agree, alpha), mixture(p, agree)
    pd_, md = product_of_experts(p, disagree, alpha), mixture(p, disagree)
    add_text(slide, Inches(8.3), Inches(1.7), Inches(4.45), Inches(3.4), [
        ("Illustrative, α = " + r.num(alpha, 0), SUB, True, ACCENT),
        (f"p = ({', '.join(r.illus(x) for x in p)})", SUB, False, INK),
        (f"agreeing q: product top {r.illus(pa[0])}, mixture {r.illus(ma[0])}", SUB, False, INK),
        (f"q = ({', '.join(r.illus(x) for x in disagree)})", SUB, False, INK),
        (f"product → ({', '.join(r.illus(x) for x in pd_)})", SUB, False, INK),
        (f"mixture → ({', '.join(r.illus(x) for x in md)})", SUB, False, INK),
    ], spacing=6)
    add_bullets(slide, [
        "Motive: TDA's product turns agreement into confidence, and the cache agrees with CLIP by construction.",
        "Why it should work: a mixture is bounded by its most confident input. Agreement cannot "
        "manufacture confidence neither source had.",
        "Why not a normalized product with a smaller α? That is still TDA; it only rescales the sharpening.",
    ], top=Inches(4.45), spacing=8)
    set_notes(slide, f"""
The second change attacks the product of experts. The table contrasts the two rules. TDA
multiplies the distributions; when they agree the result is sharper than either, and there
is no upper bound. We instead average them, with weight {wgt}. A mixture of two distributions can
never be more confident than the more confident of the two.

In the illustrative example with agreeing experts, the product gives {r.illus(pa[0])} and the mixture
stays at {r.illus(ma[0])}. When they disagree, the product lets the more peaked one win, while the
mixture produces a compromise.

You might ask why not keep a normalized product but with a smaller alpha. That is simply TDA
with a different alpha; it still sharpens on agreement. The mixture is a different kind of
combination, and that is the point.
""")
    return slide


def slide_fusion_result(prs, r):
    slide = new_slide(prs, "Change 2 · Result: fusion makes every dataset underconfident", kicker="4 · Our proposal")
    rows = []
    for d in DATASET_ORDER:
        rows.append([DATASET_LABEL[d], r.num(r.v(d, "none", "accuracy")), r.num(r.v(d, "fusion", "accuracy")),
                     r.num(r.r("ablation", d, "none", "gap"), 1, sign=True),
                     r.num(r.r("ablation", d, "fusion", "gap"), 1, sign=True),
                     r.num(r.v(d, "none", "ece")), r.num(r.v(d, "fusion", "ece"))])
    rows.append(["Mean", r.num(r.v_mean("none", "accuracy")), r.num(r.v_mean("fusion", "accuracy")),
                 r.num(r.r_mean("ablation", "none", "gap"), 1, sign=True),
                 r.num(r.r_mean("ablation", "fusion", "gap"), 1, sign=True),
                 r.num(r.v_mean("none", "ece")), r.num(r.v_mean("fusion", "ece"))])
    add_table(slide, ["Dataset", "acc none", "acc fusion", "gap none", "gap fusion", "ECE none", "ECE fusion"],
              rows, MARGIN, Inches(1.65), Inches(7.6), col_widths=[1.4, 1, 1.05, 1, 1.05, 1, 1.05],
              font_size=15, row_height=Inches(0.42), highlight_rows=(5,))
    beta_air = float(r.c("aircraft", "beta"))
    cap = int(r.c("aircraft", "pos_capacity"))
    c_typ = cap * math.exp(-beta_air * (1 - ILLUSTRATIVE["scale_cos"]))
    q_top = flat_cache_top_probability(int(r.c("aircraft", "n_classes")), c_typ)
    fused_top = 0.5 * ILLUSTRATIVE["peaked_clip_top"] + 0.5 * q_top
    add_text(slide, Inches(8.5), Inches(1.65), Inches(4.25), Inches(4.4), [
        ("Why: softmax is not scale-invariant", SUB, True, ACCENT),
        (f"CLIP's logits carry the ×{r.num(r.litv('clip_logit_scale'), 0)} scale: peaked.", SUB, False, INK),
        (f"Cache logits are sums of ≤ {r.num(cap, 0)} votes ≤ 1: nearly flat after softmax.", SUB, False, INK),
        (f"Illustrative, Aircraft: {r.num(cap, 0)} votes at cos {r.illus(ILLUSTRATIVE['scale_cos'])} → c = "
         f"{r.illus(c_typ)}, top q = {r.illus(q_top, 3)} over {r.num(r.c('aircraft', 'n_classes'), 0)} classes.",
         SUB, False, INK),
        (f"Mixed 50/50 with a CLIP top of {r.illus(ILLUSTRATIVE['peaked_clip_top'])}: "
         f"{r.illus(fused_top, 3)}. Right or wrong.", SUB, False, INK),
    ], spacing=6)
    add_placeholder(slide, MARGIN, Inches(4.72), Inches(3.7), Inches(1.38), "A6",
                    "the scale gap, measured",
                    "Histogram of max softmax(c) vs max softmax(z), per dataset.")
    add_placeholder(slide, MARGIN + Inches(3.9), Inches(4.72), Inches(3.7), Inches(1.38), "A7",
                    "dilution on CLIP-confident images",
                    "Reliability of the CLIP-confident subgroup before vs after fusion.")
    add_callout(slide, f"Mean ECE {r.num(r.v_mean('none', 'ece'))} → {r.num(r.v_mean('fusion', 'ece'))}: the worst "
                "single change. Accuracy barely moves; the confidence is halved.", color=WARN, top=Inches(6.2))
    set_notes(slide, f"""
The result is bad, and instructive. Accuracy barely moves, {r.num(r.v_mean('none', 'accuracy'))} to {r.num(r.v_mean('fusion', 'accuracy'))}, but every dataset
becomes underconfident: the gap column goes negative everywhere, by up to {r.num(-r.r('ablation', 'pets', 'fusion', 'gap'), 1)} points on Pets. Mean
ECE more than doubles.

The reason is that softmax is not scale-invariant. CLIP's logits are a hundred times a
cosine, so its softmax is peaked. The cache logits are sums of at most {r.num(cap, 0)} votes, each at
most one, so their softmax over many classes is almost flat. In the illustrative Aircraft
case, three votes at cosine {r.illus(ILLUSTRATIVE['scale_cos'])} give a cache logit of {r.illus(c_typ)}, which over a hundred classes is a
top probability of only {r.illus(q_top, 3)}. Averaging that with a confident CLIP prediction of {r.illus(ILLUSTRATIVE['peaked_clip_top'])}
gives about {r.illus(fused_top, 3)}, whether CLIP was right or wrong. The mixture is bounded, as promised, but
it is dragged down by a flat expert.

Evidence A6 will measure the scale gap directly, and A7 the dilution on images CLIP was
already confident about.
""")
    return slide


def slide_gate_rule(prs, r):
    slide = new_slide(prs, "Change 3 · Gated fusion: consult the cache only on a two-way tie", kicker="4 · Our proposal")
    max_gap = r.num(r.gate_cfg["fusion_gate_max_gap"], 1)
    min_mass = r.num(r.gate_cfg["fusion_gate_min_mass"], 1)
    add_panel(slide, MARGIN, Inches(1.65), Inches(6.9), Inches(1.75))
    add_equation(slide, r"\mathrm{tie} \Leftrightarrow p_{(1)} - p_{(2)} < g \ \ \mathrm{and}\ \ p_{(1)} + p_{(2)} > m",
                 MARGIN, Inches(1.78), size=22, center_width=Inches(6.9))
    add_equation(slide, r"\mathrm{output} = \frac{1}{2}(p+q)\ \mathrm{on\ a\ tie,}\quad p\ \mathrm{otherwise}",
                 MARGIN, Inches(2.6), size=22, center_width=Inches(6.9))
    add_bullets(slide, [
        f"g = {max_gap}, m = {min_mass}: probability space, fixed before any run.",
        "p: CLIP with the negative-cache correction. q: softmax of the positive cache.",
        "Motive: fuse only where the cache can change the answer; elsewhere it only dilutes.",
        "Second clause: spread over many classes means confused, not torn; the cache's vote is noise there.",
        "Same top-2 gap as margin admission, but used at prediction time: every class stays represented.",
    ], top=Inches(3.55), width=Inches(6.9), size=SUB, spacing=5)
    add_panel(slide, Inches(7.75), Inches(1.65), Inches(4.98), Inches(1.55))
    add_equation(slide, r"(2)\ \mathrm{overtakes}\ (1) \Leftrightarrow p_{(1)} - p_{(2)} < q_{(2)} - q_{(1)}",
                 Inches(7.75), Inches(1.72), size=20, center_width=Inches(4.98))
    add_equation(slide, r"q_{(2)} - q_{(1)} \leq \frac{e^{K} - 1}{e^{K} + C - 1}",
                 Inches(7.75), Inches(2.3), size=22, center_width=Inches(4.98))
    rows = [[DATASET_LABEL[d], r.num(r.c(d, "n_classes"), 0), r.num(r.c(d, "mixture_overturn_gap"))]
            for d in DATASET_ORDER]
    add_table(slide, ["Dataset", "C", "max gap the mixture can flip"], rows, Inches(7.75), Inches(3.35),
              Inches(4.98), col_widths=[1.4, 0.6, 2.6], font_size=15, row_height=Inches(0.37))
    safe = [DATASET_LABEL[d] for d in DATASET_ORDER
            if r.c(d, "mixture_overturn_gap") < r.gate_cfg["fusion_gate_max_gap"]]
    unsafe = [DATASET_LABEL[d] for d in DATASET_ORDER
              if r.c(d, "mixture_overturn_gap") >= r.gate_cfg["fusion_gate_max_gap"]]
    add_text(slide, Inches(7.75), Inches(5.65), Inches(4.98), Inches(1.1), [
        (f"Gap ≥ {max_gap} is provably a clear winner on {' and '.join(safe)}; "
         f"not on {', '.join(unsafe)}.", SUB, False, INK)])
    set_notes(slide, f"""
The diagnosis of the fusion failure suggests the repair. Fusion only helps where the cache
can change the answer. Everywhere else it only dilutes. So we fuse only on a two-way tie,
and otherwise output CLIP's own distribution, still with TDA's negative-cache correction.

A tie means the top two CLIP probabilities are within {max_gap} of each other, and together hold more
than {min_mass} of the mass. The second clause excludes images spread thinly over many classes: there
CLIP is confused, not torn, and the cache vote is noise. Both thresholds are in probability
space, so they mean the same on ten and a hundred classes, and they were fixed in advance.

Why is "clear winner" safe to leave alone? Under the mixture, the cache flips the answer only if
its own preference for the runner-up exceeds CLIP's gap. Cache logits are between zero and the
capacity, which bounds that preference, as in the formula. On the hundred-class datasets the
bound is {r.num(r.c('aircraft', 'mixture_overturn_gap'))}, so a CLIP gap of {max_gap} or more can never be overturned there; leaving
those images to CLIP loses no flip the mixture could have made. On EuroSAT, with ten classes, the bound is {r.num(r.c('eurosat', 'mixture_overturn_gap'))}: the gate does give up
flips there, and we will see it in the accuracy.

Note this is the same quantity as margin admission, used the opposite way: to decide when the
cache speaks, not what it remembers.
""")
    return slide


def slide_gate_result(prs, r):
    slide = new_slide(prs, "Change 3 · Result: gated fusion", kicker="4 · Our proposal")
    g = "ablation_gate"
    rows = []
    for d in DATASET_ORDER:
        rows.append([DATASET_LABEL[d], r.num(r.v(d, "none", "accuracy", g)), r.num(r.v(d, "gate", "accuracy", g)),
                     r.num(r.r("headline", d, "clipzs", "mean_confidence"), 1),
                     r.num(r.r(g, d, "gate", "mean_confidence"), 1),
                     r.num(r.s(d, "clipzs", "ece")), r.num(r.v(d, "none", "ece", g)), r.num(r.v(d, "gate", "ece", g))])
    rows.append(["Mean", r.num(r.v_mean("none", "accuracy", g)), r.num(r.v_mean("gate", "accuracy", g)),
                 r.num(r.r_mean("headline", "clipzs", "mean_confidence"), 1),
                 r.num(r.r_mean(g, "gate", "mean_confidence"), 1),
                 r.num(r.s_mean("clipzs", "ece")), r.num(r.v_mean("none", "ece", g)), r.num(r.v_mean("gate", "ece", g))])
    add_table(slide, ["Dataset", "acc none", "acc gate", "conf CLIP", "conf gate", "ECE CLIP", "ECE none", "ECE gate"],
              rows, MARGIN, Inches(1.65), Inches(8.2), col_widths=[1.4, 1, 1, 1, 1, 1, 1, 1], font_size=15,
              row_height=Inches(0.42), highlight_rows=(5,))
    beats = [d for d in DATASET_ORDER if r.v(d, "gate", "ece", g) < r.s(d, "clipzs", "ece")]
    loses = [d for d in DATASET_ORDER if d not in beats]
    drop_eu = r.v("eurosat", "none", "accuracy", g) - r.v("eurosat", "gate", "accuracy", g)
    add_bullets(slide, [
        f"Mean ECE {r.num(r.v_mean('none', 'ece', g))} → {r.num(r.v_mean('gate', 'ece', g))}, below CLIP's mean.",
        f"Beats CLIP's ECE on {r.num(len(beats), 0)} of {r.num(len(DATASET_ORDER), 0)} datasets; worse on "
        f"{' and '.join(DATASET_LABEL[d] for d in loses)}.",
        "Mean confidence tracks CLIP's on every dataset.",
        f"Cost: accuracy, most on EuroSAT (−{r.num(drop_eu)}), as the flip bound predicts.",
    ], left=Inches(9.05), top=Inches(1.65), width=Inches(3.7), size=SUB, spacing=8)
    add_placeholder(slide, MARGIN, Inches(4.75), Inches(8.2), Inches(1.25), "A8",
                    "how often the gate fires, and per-branch calibration",
                    "Firing rate per dataset; reliability of tie rows vs non-tie rows.")
    add_callout(slide, "Disclosure: the gate was chosen among four fusion variants evaluated on the test sets "
                "(backup B2). P2 and val selection will check it.", color=BASELINE, top=Inches(6.2))
    set_notes(slide, f"""
With the gate, the blow-up is gone. Mean ECE drops from {r.num(r.v_mean('none', 'ece', g))} for the baseline to {r.num(r.v_mean('gate', 'ece', g))}, which is
below CLIP's own mean of {r.num(r.s_mean('clipzs', 'ece'))}. But I want to be precise: it beats CLIP's ECE on {r.num(len(beats), 0)} of 5 datasets, not
everywhere; on {' and '.join(DATASET_LABEL[d] for d in loses)} it is worse than CLIP.

The confidence columns show what the gate does: its mean confidence tracks CLIP's on every
dataset. In other words it removes TDA's inflation and returns roughly to CLIP's own
calibration, while keeping part of TDA's accuracy.

The cost is accuracy, mostly on EuroSAT, minus {r.num(drop_eu)} points, which is what the flip bound on
the previous slide predicted: on ten classes the cache could usefully overturn larger gaps,
and the gate forbids that.

And a disclosure: during exploration we compared four fusion variants on the test sets and
picked this one. That is selection on test. The backup slide lists all four, and the clean
protocol and validation-split selection will check whether the choice holds.
""")
    return slide


def slide_temp_motive(prs, r):
    slide = new_slide(prs, "Change 4 · LOO temperature: motive and how it works", kicker="4 · Our proposal")
    lo, hi = r.num(r.grid[0], 1), r.num(r.grid[-1], 1)
    step = r.num(r.grid[1] - r.grid[0])
    n_grid = r.num(len(r.grid), 0)
    add_panel(slide, MARGIN, Inches(1.65), Inches(7.1), Inches(2.05))
    add_equation(slide, r"\hat{a} = \frac{1}{M}\sum_{i=1}^{M} \mathbf{1}\!\left[\hat{y}_{\mathrm{nn}(i)} = \hat{y}_i\right]",
                 MARGIN, Inches(1.7), size=22, center_width=Inches(7.1))
    add_equation(slide, r"T^{\star} = \arg\min_{T \in \mathcal{G}} \left|\,\overline{\mathrm{conf}}(T) - \hat{a}\,\right|,"
                 r"\qquad p = \mathrm{softmax}(z / T^{\star})",
                 MARGIN, Inches(2.85), size=22, center_width=Inches(7.1))
    add_bullets(slide, [
        "Motive: TDA's error is a near-uniform shift (ECE ≈ gap), the textbook case for one scalar T.",
        "No labels: â = leave-one-out 1-NN agreement of the cached pseudo-labels; conf(T) = mean "
        "confidence over past images.",
        f"Grid 𝒢 = {lo}, …, {hi} in steps of {step} ({n_grid} values). Floor T = {lo}: it can only soften.",
        "Rank-preserving: accuracy cannot change.",
    ], top=Inches(3.85), width=Inches(7.1), size=SUB, spacing=5)
    add_text(slide, Inches(8.05), Inches(1.65), Inches(4.68), Inches(2.6), [
        ("Cost, as implemented", SUB, True, WARN),
        (f"Every step re-scores the whole history for all {n_grid} temperatures: O(N²) over a stream of N images.",
         SUB, False, INK),
        ("Task T03 keeps running sums per grid value: O(N), identical temperatures.", SUB, False, INK),
    ], spacing=6)
    add_placeholder(slide, Inches(8.05), Inches(4.35), Inches(4.68), Inches(1.65), "T06 timing",
                    "per-image CPU time", "CLIP, TDA, gate, temperature (old and O(N)) on the same CPU.")
    add_text(slide, MARGIN, Inches(6.5), CONTENT_W, Inches(0.4), [
        ("Prior art: UL-TTA (arXiv'25) also sets a label-free test-time temperature; ours targets the "
         "cache's own LOO agreement.", SMALL, False, MUTED)])
    set_notes(slide, f"""
The fourth change. The calibration section showed that TDA's error is close to a uniform
upward shift: ECE about equal to the gap. That is the textbook case for temperature
scaling, dividing all logits by one number. Normally the temperature is fitted on labeled
validation data, which we do not have.

So we estimate accuracy without labels: for each cached item, find its nearest other cached
item and check whether their pseudo-labels agree. The fraction that agree is our estimate.
Then we pick the temperature whose mean confidence over the history matches it, from a
grid that starts at {lo}, so it can only soften, never sharpen. Since it is a monotone rescaling,
accuracy cannot change.

Cost: as implemented, every step re-scores the full history for all {n_grid} grid values, which is
quadratic in the stream length. That is an implementation artifact; task T03 makes it linear
with identical temperatures, and we will time both.

UL-TTA, a recent paper, also sets a temperature without labels; we cite it as prior art.
""")
    return slide


def slide_temp_result(prs, r):
    slide = new_slide(prs, "Change 4 · Result: helps where TDA is overconfident, hurts where not",
                      kicker="4 · Our proposal")
    rows = []
    for d in DATASET_ORDER:
        rows.append([DATASET_LABEL[d], r.num(r.r("ablation", d, "none", "gap"), 1, sign=True),
                     r.num(r.r("ablation", d, "temp", "gap"), 1, sign=True),
                     r.num(r.v(d, "none", "ece")), r.num(r.v(d, "temp", "ece")),
                     r.num(r.s(d, "cal_tda_temp", "temp_mean")),
                     r.num(100 * r.s(d, "cal_tda_temp", "temp_frac_at_grid_boundary"), 1) + "%"])
    rows.append(["Mean", r.num(r.r_mean("ablation", "none", "gap"), 1, sign=True),
                 r.num(r.r_mean("ablation", "temp", "gap"), 1, sign=True),
                 r.num(r.v_mean("none", "ece")), r.num(r.v_mean("temp", "ece")), "", ""])
    add_table(slide, ["Dataset", "gap none", "gap temp", "ECE none", "ECE temp", "mean T", "on grid edge"],
              rows, MARGIN, Inches(1.65), Inches(8.2), col_widths=[1.4, 1, 1, 1, 1, 0.9, 1.3], font_size=15,
              row_height=Inches(0.42), highlight_rows=(5,))
    same = all(abs(r.v(d, "none", "accuracy") - r.v(d, "temp", "accuracy")) < 1e-9 for d in DATASET_ORDER)
    add_bullets(slide, [
        "Accuracy identical on every dataset, as it must be." if same else "Accuracy changed: a bug.",
        "Big wins where TDA was most overconfident (DTD, Aircraft).",
        f"EuroSAT stays overconfident: T sits at the floor on most steps; â is too high.",
        f"Pets made underconfident (mean T {r.num(r.s('pets', 'cal_tda_temp', 'temp_mean'))}): â is too low "
        "(likely: similar breeds' neighbours disagree even when correct; A9 checks).",
    ], left=Inches(9.05), top=Inches(1.65), width=Inches(3.7), size=SUB, spacing=6)
    add_placeholder(slide, MARGIN, Inches(4.75), Inches(8.2), Inches(1.25), "A9",
                    "estimator quality",
                    "â vs the true running accuracy over the stream; estimator error; oracle-T ceiling (labels).")
    add_callout(slide, "Keep or drop? Optional add-on for additive TDA only; it does not stack on the gate (next) "
                "and is not in the recommended method.", color=BASELINE, top=Inches(6.2))
    set_notes(slide, f"""
Results for the temperature alone. Accuracy is identical on every dataset, as it must be. Mean
ECE improves from {r.num(r.v_mean('none', 'ece'))} to {r.num(r.v_mean('temp', 'ece'))}, with large wins on DTD and Aircraft, where TDA was most
overconfident.

But the estimate is wrong in both directions. On EuroSAT it is too high: the search wants to
sharpen, the floor holds it at one, and the temperature sits on the grid edge on
{r.num(100 * r.s('eurosat', 'cal_tda_temp', 'temp_frac_at_grid_boundary'), 1)} percent of steps,
so EuroSAT stays overconfident. On Pets it is too low, and the search softens a model that
was already nearly calibrated. Our hypothesis is that neighbouring breeds look alike, so a
correctly labeled cached Birman's nearest neighbour is often a correctly labeled Ragdoll and
the agreement understates accuracy. Evidence A9 will compare the estimate with the true
running accuracy and test this.

Should we keep it? Given the cost and that it does not combine with the gate, we keep it only
as an optional add-on for plain TDA, not in our recommended method.
""")
    return slide


# --- Results ------------------------------------------------------------------------------
HEADLINE = [("clipzs", "CLIP zero-shot"), ("tda", "TDA"), ("cal_tda_gate", "Ours: gated fusion"),
            ("cal_tda_temp", "Ours: LOO temperature"), ("cal_tda_gate_temp", "Ours: gate + temperature")]


def slide_headline(prs, r):
    slide = new_slide(prs, "Headline results (protocol P1, seed 0)", kicker="5 · Results")
    for block, (column, title) in enumerate((("accuracy", "Top-1 accuracy (%), higher is better"),
                                             ("ece", "ECE (%), lower is better"))):
        top = Inches(1.55) + block * Inches(2.42)
        add_text(slide, MARGIN, top, CONTENT_W, Inches(0.35), [(title, SUB, True, MUTED)])
        rows = [[label] + [r.num(r.s(d, m, column)) for d in DATASET_ORDER] + [r.num(r.s_mean(m, column))]
                for m, label in HEADLINE]
        add_table(slide, ["Method"] + [DATASET_LABEL[d] for d in DATASET_ORDER] + ["Mean"], rows, MARGIN,
                  top + Inches(0.36), CONTENT_W, col_widths=[2.6, 1, 1.1, 1, 1, 1, 1], font_size=14,
                  row_height=Inches(0.33), highlight_rows=(2,))
    kept = r.s_mean("cal_tda_gate", "accuracy") - r.s_mean("clipzs", "accuracy")
    gain = r.s_mean("tda", "accuracy") - r.s_mean("clipzs", "accuracy")
    lowest = min(HEADLINE, key=lambda mm: r.s_mean(mm[0], "ece"))[0] == "cal_tda_gate"
    add_callout(slide, f"Gated fusion keeps {r.num(kept)} of TDA's {r.num(gain)} accuracy points over CLIP"
                + (f" and has the lowest mean ECE ({r.num(r.s_mean('cal_tda_gate', 'ece'))})." if lowest else "."),
                color=GOOD, top=Inches(6.42), height=Inches(0.5))
    set_notes(slide, f"""
The headline table, protocol P1, seed zero. Top: accuracy; bottom: ECE. The highlighted row is
our recommendation, gated fusion.

On average, CLIP reaches {r.num(r.s_mean('clipzs', 'accuracy'))} percent at {r.num(r.s_mean('clipzs', 'ece'))} ECE, TDA {r.num(r.s_mean('tda', 'accuracy'))} at {r.num(r.s_mean('tda', 'ece'))}. Gated fusion reaches
{r.num(r.s_mean('cal_tda_gate', 'accuracy'))} at {r.num(r.s_mean('cal_tda_gate', 'ece'))}. So it keeps {r.num(kept)} of TDA's {r.num(gain)} points of accuracy gain and has the lowest mean ECE
of all five, although, as said, not the lowest on every dataset.

The temperature row shows unchanged accuracy and better ECE than TDA except on Pets, and the
combination is not better than the gate alone. That is the ablation, two slides on.
""")
    return slide


def slide_protocols(prs, r):
    slide = new_slide(prs, "Is it fair? Two hyperparameter protocols", kicker="5 · Results")
    rows = [[DATASET_LABEL[d], r.num(r.c(d, "alpha"), 1), r.num(r.c(d, "beta"), 1), r.num(r.c(d, "p2_alpha"), 1),
             r.num(r.c(d, "p2_beta"), 1)] for d in DATASET_ORDER]
    add_table(slide, ["Dataset", "P1 α", "P1 β", "P2 α", "P2 β"], rows, MARGIN, Inches(1.65), Inches(5.2),
              col_widths=[1.5, 0.8, 0.8, 0.8, 0.8], font_size=16, row_height=Inches(0.44))
    add_bullets(slide, [
        "P1 “published”: TDA's per-dataset values from its repo. Comparable with the paper and "
        "the benchmark; provenance unstated.",
        "P2 “clean”: one setting (the paper's own) on every dataset, for every method. No method has "
        "seen any of the five test sets.",
        "Our knobs (margin threshold, w, gate thresholds, grid floor) are global and fixed a priori.",
        "Rule (agreed): P2 is the headline if P2-TDA beats CLIP's accuracy; both are always shown.",
    ], left=Inches(6.15), top=Inches(1.65), width=Inches(6.6), size=SUB, spacing=8)
    add_placeholder(slide, MARGIN, Inches(4.5), Inches(5.2), Inches(1.6), "T07",
                    "P1 vs P2 results, seeds 0/1/2",
                    "Accuracy, ECE, gap, mean ± std for CLIP, TDA and our variants.")
    add_callout(slide, "Answers “is it fair?”: no method gets per-dataset tuning that the others do not, and P2 "
                "uses none at all.", top=Inches(6.2))
    set_notes(slide, f"""
Is the comparison fair? Two issues. TDA's per-dataset alpha and beta come from its repository
with no stated selection procedure, so they may encode test-set information. And our own
knobs could have been tuned.

We therefore report two protocols. P1 uses the published per-dataset values, which makes our
numbers comparable with the paper and the benchmark. P2 uses one setting everywhere, alpha {r.num(r.c('dtd', 'p2_alpha'), 0)}
and beta {r.num(r.c('dtd', 'p2_beta'), 0)}, the paper's own searched value, for every method; under P2 nothing has seen any
of our test sets. Our own knobs are global constants fixed in advance in both.

We agreed a decision rule beforehand: if TDA under P2 still beats CLIP's accuracy, P2 becomes
the headline. The P2 numbers, with three seeds, are pending, task T07.
""")
    return slide


def slide_ablation(prs, r):
    slide = new_slide(prs, "Ablation: gate and temperature do not stack", kicker="5 · Results")
    g = "ablation_gate"
    labels = [("none", "neither (TDA through our code)"), ("gate", "gated fusion"),
              ("temp", "LOO temperature"), ("gate_temp", "both")]
    rows = [[label, r.num(r.v_mean(v, "accuracy", g)), r.num(r.v_mean(v, "ece", g)),
             r.num(r.r_mean(g, v, "gap"), 1, sign=True)] for v, label in labels]
    add_table(slide, ["Variant (margin off), mean of 5", "Accuracy", "ECE", "gap"], rows, MARGIN, Inches(1.65),
              Inches(6.0), col_widths=[3.0, 1.1, 0.9, 0.9], font_size=16, row_height=Inches(0.46), highlight_rows=(1,))
    eu = 100 * r.s("eurosat", "cal_tda_gate_temp", "temp_frac_at_grid_boundary")
    air = 100 * r.s("aircraft", "cal_tda_gate_temp", "temp_frac_at_grid_boundary")
    add_bullets(slide, [
        f"Adding temperature to the gate: accuracy unchanged, ECE {r.num(r.v_mean('gate', 'ece', g))} → "
        f"{r.num(r.v_mean('gate_temp', 'ece', g))}.",
        "Both remove the same overconfidence; after the gate, what is left is CLIP's own calibration.",
        f"On the gated output the search wants T < 1 and the floor blocks it: on a grid edge for "
        f"{r.num(eu, 1)}% of EuroSAT and {r.num(air, 1)}% of Aircraft steps.",
        f"On Pets it pushes the wrong way ({r.num(r.v('pets', 'gate', 'ece', g))} → "
        f"{r.num(r.v('pets', 'gate_temp', 'ece', g))}).",
    ], left=Inches(6.95), top=Inches(1.65), width=Inches(5.78), size=SUB, spacing=8)
    per = [[DATASET_LABEL[d], r.num(r.v(d, "gate", "ece", g)), r.num(r.v(d, "gate_temp", "ece", g)),
            r.num(100 * r.s(d, "cal_tda_gate_temp", "temp_frac_at_grid_boundary"), 1) + "%"] for d in DATASET_ORDER]
    add_table(slide, ["Dataset", "ECE gate", "gate + temp", "T on grid edge"], per, MARGIN, Inches(4.2),
              Inches(6.0), col_widths=[1.5, 1.1, 1.2, 1.4], font_size=15, row_height=Inches(0.36))
    add_callout(slide, "Recommendation: gated fusion alone.", color=GOOD, top=Inches(5.95), left=Inches(6.95),
                width=Inches(5.78), height=Inches(0.55))
    set_notes(slide, f"""
The two surviving changes, in all four on/off combinations, margin admission off.

The gate alone gives the best ECE. Adding temperature on top leaves accuracy unchanged, as it
must, and makes ECE slightly worse, {r.num(r.v_mean('gate', 'ece', g))} to {r.num(r.v_mean('gate_temp', 'ece', g))}. The reason is that they fix the same
problem: the gate already removes TDA's inflation, and what is left is about CLIP's own
calibration. On EuroSAT and Aircraft the temperature search then wants to sharpen, the floor
blocks it, and it does nothing on {r.num(eu, 1)} and {r.num(air, 1)} percent of steps; on Pets it softens in the wrong
direction.

The lower table shows it per dataset. So we recommend the gate alone.
""")
    return slide


def slide_baselines(prs, r):
    slide = new_slide(prs, "Baselines and positioning", kicker="5 · Results")
    shots = r.num(r.litv("tip_adapter_shots"), 0)
    rows = [
        ["ECALP (ICLR'25)", "zero-shot, online", "run by us (replay)", "most accurate online method, worst calibrated"],
        ["DMN-weak (CVPR'24)", "zero-shot, online", "run by us (replay)", "widely used memory method"],
        ["Tip-Adapter", f"few-shot ({shots}-shot)", "run by us", "what labels buy: reference, not a competitor"],
        ["TPT, C-TPT, O-TPT, D-TPT", "zero-shot, episodic", "cited from papers", "calibration-aware; need backprop"],
    ]
    add_table(slide, ["Method", "Setting", "How", "Why"], rows, MARGIN, Inches(1.65), CONTENT_W,
              col_widths=[2.6, 2.0, 2.0, 4.4], font_size=16, row_height=Inches(0.5), align=PP_ALIGN.LEFT,
              bold_first_col=True)
    add_placeholder(slide, MARGIN, Inches(4.35), CONTENT_W, Inches(1.65), "T08",
                    "baseline table: accuracy, ECE, gap on the five datasets",
                    "ECALP and DMN-weak with their published settings (P1 only, noted); Tip-Adapter "
                    f"{shots}-shot with α, β chosen on val; cited ECE with protocol differences stated.")
    add_callout(slide, "Answers “why only TDA?”: two more zero-shot methods run, one few-shot reference, "
                "calibration-specific methods cited.", top=Inches(6.2))
    set_notes(slide, f"""
To answer "why only TDA", we will add baselines. Two more zero-shot online methods run by us
from the cached features: ECALP, the most accurate online method in the benchmark and also
the worst calibrated, and DMN in its zero-shot variant. One few-shot reference, Tip-Adapter
with {shots} labeled images per class, to show what labels buy; it is not a competitor in our
setting. And the calibration-aware prompt-tuning methods, which need backprop and many views,
are cited from their papers with the protocol differences stated.

This table is pending, task T08.
""")
    return slide


def slide_limitations(prs, r):
    slide = new_slide(prs, "Limitations and next steps", kicker="6 · Limitations and conclusion")
    seeds = sorted(set(r.summary["seed"]))
    fs_flowers = r.r("exploration", "flower102", "fusion_sharpen", "accuracy")
    add_bullets(slide, [
        f"One seed ({r.num(seeds[0], 0)}) and one backbone (ViT-B/16). Next: seeds 0/1/2 by feature replay (T07).",
        "Selection history: the gate was chosen among four fusion variants evaluated on the test sets; "
        f"one was never run on EuroSAT, and another reached {r.num(fs_flowers)} on Flowers102 "
        f"(TDA {r.num(r.s('flower102', 'tda', 'accuracy'))}). Next: P2 and val-split selection.",
        "Per-dataset α/β of unstated provenance (P1). Next: P2 everywhere, and TDA's own ablations with "
        "ECE (T06).",
        "LOO estimate quality: wrong in both directions (EuroSAT too high, Pets too low). Next: A9.",
        "Mechanism claims rest on math and aggregate statistics so far. Next: traces A2–A8.",
        "Temperature is O(N²) as implemented (T03). Calibration-specific baselines are cited, not run.",
    ], spacing=10, size=SUB)
    set_notes(slide, f"""
What we cannot claim yet. One seed and one backbone; replaying cached features will make three
seeds cheap.

Selection history: the gated variant was picked among four fusion variants looked at on the
test sets. One of them was never run on EuroSAT, and another one actually beat TDA's accuracy on
Flowers, {r.num(fs_flowers)} against {r.num(r.s('flower102', 'tda', 'accuracy'))}. We disclose this, and the clean protocol plus selection on the
validation split will test whether the choice survives.

TDA's own per-dataset hyperparameters have unstated provenance; protocol P2 removes them.

The temperature estimate is unreliable in both directions, and the mechanism claims so far are
supported by math and aggregate statistics; the trace analyses will test them directly.
""")
    return slide


def slide_conclusion(prs, r):
    slide = new_slide(prs, "Conclusion", kicker="6 · Limitations and conclusion")
    add_panel(slide, MARGIN, Inches(1.65), CONTENT_W, Inches(1.45), fill=TINT_ACCENT, line=ACCENT)
    add_text(slide, MARGIN + Inches(0.3), Inches(1.8), CONTENT_W - Inches(0.6), Inches(1.2), [
        ("What we recommend", SUB, True, ACCENT),
        (f"TDA's cache with gated fusion: mean ECE {r.num(r.s_mean('tda', 'ece'))} → {r.num(r.s_mean('cal_tda_gate', 'ece'))} "
         f"(CLIP {r.num(r.s_mean('clipzs', 'ece'))}) at {r.num(r.s_mean('cal_tda_gate', 'accuracy'))}% accuracy "
         f"(TDA {r.num(r.s_mean('tda', 'accuracy'))}, CLIP {r.num(r.s_mean('clipzs', 'accuracy'))}). "
         "One extra comparison per image.", 22, True, INK),
    ], spacing=4)
    add_bullets(slide, [
        "TDA's overconfidence comes from its cache adding confidence regardless of correctness: "
        "self-votes and a product of experts that agree by construction.",
        "Filtering the cache by margin fails; mixing instead of multiplying fails through a scale mismatch; "
        "consulting the cache only on two-way ties works.",
        "A label-free temperature helps plain TDA but adds nothing after the gate.",
        "Pending: P2 protocol, three seeds, mechanism traces, and baselines.",
    ], top=Inches(3.35), spacing=10)
    set_notes(slide, f"""
To conclude. TDA's overconfidence is structural: the cache adds confidence regardless of
correctness, through self-votes and a product of experts that agree by construction.

Of our four changes, filtering the cache by margin fails, and mixing instead of multiplying
fails because of a scale mismatch. Consulting the cache only where CLIP is torn between two
classes works: mean ECE from {r.num(r.s_mean('tda', 'ece'))} to {r.num(r.s_mean('cal_tda_gate', 'ece'))}, at {r.num(r.s_mean('cal_tda_gate', 'accuracy'))} percent accuracy, for one comparison per image.
The temperature helps plain TDA but adds nothing on top of the gate.

What remains is to confirm this under the clean protocol, over three seeds, with direct traces of
the mechanism and against other methods. Thank you.
""")
    return slide


# --- Backup -----------------------------------------------------------------------------
def slide_b_tda_ablations(prs, r):
    slide = new_slide(prs, "Backup B1 · TDA's own ablations, redone with ECE", kicker="Backup")
    add_bullets(slide, [
        "Fig. 3: CLIP vs positive-only vs negative-only vs TDA: which cache causes the overconfidence?",
        "Fig. 4: shot capacity 1–6: purity vs coverage; does calibration follow accuracy?",
        "Fig. 5 and Table 5: negative-cache mask threshold and entropy band.",
        "Table 6: α × β heatmaps per dataset, accuracy and ECE, P1 and P2 points marked.",
    ], width=CONTENT_W, spacing=8, size=SUB)
    add_placeholder(slide, MARGIN, Inches(3.75), CONTENT_W, Inches(2.2), "T06",
                    "TDA paper figures with ECE",
                    "Each of TDA's ablations on our five datasets, accuracy plus ECE and gap, labelled as test-split "
                    "sensitivity analyses.")
    set_notes(slide, """
Backup. We will redo TDA's own ablations with calibration added: which of the two caches
causes the overconfidence, how the shot capacity trades purity against coverage, the negative
cache thresholds, and alpha-beta heatmaps with both protocols marked. Test-split sweeps here are
sensitivity analyses only; no choice is made from them.
""")
    return slide


def slide_b_selection(prs, r):
    slide = new_slide(prs, "Backup B2 · Selection history: the four fusion variants (test sets)", kicker="Backup")
    variants = [("fusion_gate", "gated (chosen)"), ("fusion_gate_sharpen", "gated + range-matched cache"),
                ("fusion_gate_temp", "gated + temperature"), ("fusion_sharpen", "range-matched cache, ungated")]
    rows = []
    for v, label in variants:
        row = [label]
        for d in DATASET_ORDER:
            try:
                row.append(f"{r.num(r.r('exploration', d, v, 'accuracy'), 1)} / {r.num(r.r('exploration', d, v, 'ece'), 1)}")
            except KeyError:
                row.append("not run")
        rows.append(row)
    rows.append(["TDA (reference)"] + [f"{r.num(r.s(d, 'tda', 'accuracy'), 1)} / {r.num(r.s(d, 'tda', 'ece'), 1)}"
                                        for d in DATASET_ORDER])
    add_table(slide, ["Variant (acc / ECE)"] + [DATASET_LABEL[d] for d in DATASET_ORDER], rows, MARGIN, Inches(1.7),
              CONTENT_W, col_widths=[3.2, 1.2, 1.3, 1.2, 1.2, 1.2], font_size=15, row_height=Inches(0.46),
              highlight_rows=(0,))
    add_bullets(slide, [
        "All four were evaluated on the test sets during exploration; the gated variant was then adopted.",
        "Range-matched cache: cache logits rescaled to CLIP's logit range before mixing (cf. ZS-Norm).",
    ], top=Inches(4.75), size=SUB, spacing=6)
    set_notes(slide, """
For full disclosure, these are the four fusion variants we looked at on the test sets before
settling on the plain gate. One of them, the gated variant with a range-matched cache, was never
run on EuroSAT. The ungated range-matched variant beat TDA's accuracy on Flowers. Choosing among
these on test is selection on test; the clean protocol and validation-split selection will redo
the choice properly.
""")
    return slide


def slide_b_scatter(prs, r):
    slide = new_slide(prs, "Backup B3 · Accuracy vs ECE, all headline runs", kicker="Backup")
    add_picture(slide, FIGURES / "accuracy_vs_ece.png", MARGIN, Inches(1.6), CONTENT_W, Inches(5.2))
    set_notes(slide, """
The headline results as a scatter: accuracy to the right is better, ECE down is better. Each
coloured line joins one dataset's methods. The target corner is TDA's accuracy at CLIP's ECE.
""")
    return slide


def slide_b_first_ablation(prs, r):
    slide = new_slide(prs, "Backup B4 · First ablation: all eight on/off combinations", kicker="Backup")
    labels = [("none", "none"), ("margin", "margin"), ("fusion", "fusion"), ("temp", "temperature"),
              ("margin_fusion", "margin + fusion"), ("margin_temp", "margin + temperature"),
              ("fusion_temp", "fusion + temperature"), ("all", "all three")]
    rows = [[label, r.num(r.v_mean(v, "accuracy")), r.num(r.v_mean(v, "ece")),
             r.num(r.r_mean("ablation", v, "gap"), 1, sign=True)] for v, label in labels]
    add_table(slide, ["Variant, mean of 5", "Accuracy", "ECE", "gap"], rows, MARGIN, Inches(1.65), Inches(5.6),
              col_widths=[2.6, 1.1, 1.0, 0.9], font_size=15, row_height=Inches(0.44))
    better = [label for v, label in labels if v != "none" and r.v_mean(v, "ece") < r.v_mean("none", "ece")]
    add_bullets(slide, [
        f"Variants with lower mean ECE than none: {', '.join(better) or 'none'}.",
        "Every variant with margin admission or ungated fusion is worse calibrated than none, "
        "which is why margin was dropped and fusion was gated." if better == ["temperature"] else
        "See the table for which variants beat none.",
        "Ungated fusion flips the sign of the gap (underconfident); margin makes it larger.",
    ], left=Inches(6.6), top=Inches(1.65), width=Inches(6.13), size=SUB, spacing=8)
    set_notes(slide, """
The first ablation, every on/off combination of margin admission, ungated fusion and temperature.
Any variant with margin admission or ungated fusion is worse than the baseline on ECE, which is
why those were dropped or repaired.
""")
    return slide


def slide_b_implementation(prs, r):
    slide = new_slide(prs, "Backup B5 · Implementation notes", kicker="Backup")
    diffs = [abs(r.v(d, "none", "accuracy") - r.s(d, "tda", "accuracy")) for d in DATASET_ORDER]
    ece_diffs = [abs(r.v(d, "none", "ece") - r.s(d, "tda", "ece")) for d in DATASET_ORDER]
    add_bullets(slide, [
        f"“none” vs the upstream TDA run: accuracy within {r.num(max(diffs))} points and ECE within "
        f"{r.num(max(ece_diffs))} points on every dataset (separate runs, same settings).",
        "TDA's “normalized” entropy divides an entropy in nats by log₂ C, so its maximum is ln 2, "
        "not 1. The negative-cache band is applied to that quantity; we keep it as is.",
        "Pseudo-labels and entropies use raw CLIP logits; negative-cache correction is applied before "
        "our fusion and gate, positive cache after.",
        "Every number in this deck is generated from tracked CSVs by presentation/build_pptx.py; "
        "derived statistics by analysis/deck_evidence.py.",
    ], spacing=10, size=SUB)
    set_notes(slide, f"""
A few implementation notes for questions. Our "none" baseline and the separately run upstream
TDA agree within {r.num(max(diffs))} accuracy points and {r.num(max(ece_diffs))} ECE points. TDA's normalized entropy has a quirk:
nats divided by log base two of the class count, so it never reaches one; we keep it unchanged
for faithfulness. And every number in the deck is produced by code from tracked files.
""")
    return slide


def slide_b_metrics(prs, r):
    slide = new_slide(prs, "Backup B6 · Other calibration and selective-prediction metrics", kicker="Backup")
    add_bullets(slide, [
        "ECE depends on the binning; we will add proper scoring rules and a selective-prediction curve.",
        "NLL and Brier score: reward confident correct and punish confident wrong predictions.",
        "AURC: area under the risk–coverage curve, the abstention use case directly.",
    ], spacing=8, size=SUB)
    add_placeholder(slide, MARGIN, Inches(3.4), CONTENT_W, Inches(2.2), "A10",
                    "NLL, Brier and AURC for every headline run",
                    "One table, five methods by five datasets, to check that the ECE ranking is not a binning artifact.")
    set_notes(slide, """
Backup on metrics. ECE depends on binning, so we will also report negative log-likelihood,
the Brier score and the area under the risk-coverage curve, which measures the abstention use
case directly. Evidence A10.
""")
    return slide


MAIN_BUILDERS = [
    slide_title, slide_agenda,
    slide_problem, slide_clip, slide_clip_fails, slide_landscape, slide_why_tda,
    slide_tda_idea, slide_tda_pipeline, slide_tda_write, slide_tda_read, slide_tda_alpha_beta,
    slide_tda_negative, slide_tda_poe, slide_tda_example, slide_tda_gains, slide_tda_gain_source,
    slide_ece, slide_why_calibration, slide_clip_vs_tda_cal, slide_reliability, slide_anatomy,
    slide_proposal,
    slide_margin, slide_fusion_motive, slide_fusion_result, slide_gate_rule, slide_gate_result,
    slide_temp_motive, slide_temp_result,
    slide_headline, slide_protocols, slide_ablation, slide_baselines,
    slide_limitations, slide_conclusion,
]
BACKUP_BUILDERS = [slide_b_tda_ablations, slide_b_selection, slide_b_scatter, slide_b_first_ablation,
                   slide_b_implementation, slide_b_metrics]
SLIDE_BUILDERS = MAIN_BUILDERS + BACKUP_BUILDERS
TDA_SECTION = [slide_tda_idea, slide_tda_pipeline, slide_tda_write, slide_tda_read, slide_tda_alpha_beta,
               slide_tda_negative, slide_tda_poe, slide_tda_example, slide_tda_gains, slide_tda_gain_source,
               slide_anatomy]


def slide_heading(slide, index):
    if index == 1:
        return "Title"
    for shape in slide.shapes:
        if shape.name == "title":
            return shape.text_frame.text.strip()
    return f"Slide {index}"


def write_markdown_mirror(prs, md_path) -> Path:
    lines = [
        "# Calibration-Aware Test-Time Adaptation for CLIP — draft deck v2",
        "",
        "**Generated file — do not edit.** Produced by `python presentation/build_pptx.py`",
        "alongside `calibration_aware_tta.pptx`; edit the slide functions in that script instead.",
        "",
        "Each section is one slide: its on-screen heading and the speaker notes from its notes pane.",
        "",
    ]
    for index, slide in enumerate(prs.slides, start=1):
        notes = slide.notes_slide.notes_text_frame.text.strip() if slide.has_notes_slide else ""
        lines += ["---", "", f"## {index}. {slide_heading(slide, index)}", ""]
        lines += [notes, ""] if notes else []
    md_path = Path(md_path)
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return md_path


def build(out_path, markdown_path=None, root=ROOT):
    """Build the deck; returns (path, results) so tests can inspect what was emitted."""
    results = Results(root)
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    n_main = len(MAIN_BUILDERS)
    for index, builder in enumerate(SLIDE_BUILDERS, start=1):
        slide = builder(prs, results)
        if index > 1:
            label = None if index <= n_main else f"B{index - n_main}"
            add_footer(slide, index, n_main, label)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_path))
    if markdown_path:
        write_markdown_mirror(prs, markdown_path)
    return out_path, results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="presentation/calibration_aware_tta.pptx")
    parser.add_argument("--markdown", default="presentation/slides.md")
    args = parser.parse_args()
    path, _ = build(args.out, markdown_path=args.markdown or None)
    print(f"wrote {path} ({len(SLIDE_BUILDERS)} slides: {len(MAIN_BUILDERS)} main + {len(BACKUP_BUILDERS)} backup)")


if __name__ == "__main__":
    main()
