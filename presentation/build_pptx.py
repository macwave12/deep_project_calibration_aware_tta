"""Generate deck v6: agreement-gated TDA, aligned with the final report (17 main slides + 3 backup).

Diagnosis and design slides use the val splits; results slides use the test splits and the held-out near and
far tiers; every evidence slide says its split in its kicker.

Every number on a slide or in the notes is READ from tracked files at build time and formatted through
`Results.num` (or `Results.illus` for the labelled illustrative values), which records it.
`tests/test_build_pptx.py` scans every slide and note for numerals and fails on any that did not pass
through those formatters, so a hand-typed number cannot reach the deck.

Sources: the tracked CSVs that `analysis/report_assets.py` reads (same numbers as the report),
`configs/method/cal_tda_agree.yaml` (alpha, beta, shot capacity), figures from
`analysis/paper_figures.py` and `analysis/story.py`, and the TDA paper's Fig. 1 and Fig. 2
(arXiv 2403.18293 source, `presentation/assets/tda_paper/`, cited on the slides).

    python analysis/paper_figures.py && python presentation/build_pptx.py

Writes `presentation/calibration_aware_tta.pptx` (16:9) and the speaker-notes mirror `presentation/slides.md`.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

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
from analysis import no_info as ni  # noqa: E402
from analysis import paper_figures as pf  # noqa: E402
from analysis import report_assets as ra  # noqa: E402
from analysis.story import NAMES, SIGNAL_HEAD  # noqa: E402

ROOT = _ROOT
ASSETS = ROOT / "presentation" / "assets"
FIGURES = ROOT / "analysis" / "figures"
TDA_FIGS = ASSETS / "tda_paper"
CORE = ("dtd", "flower102", "pets", "eurosat", "aircraft")

# --- Theme ---------------------------------------------------------------------
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
TINT_GATE = RGBColor(0xDA, 0xEF, 0xEC)    # the gate's rows: light teal, the gate's colour in the figures
TINT_NEUTRAL = RGBColor(0xEC, 0xE9, 0xE2)  # mean rows

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


# --- Data ----------------------------------------------------------------------
class Results:
    """Every number the deck shows, read from tracked files, formatted and recorded."""

    def __init__(self, root: Path = ROOT):
        root = Path(root)
        self.d = ra.load(root)
        self.cfg = yaml.safe_load((root / "configs/method/cal_tda_agree.yaml").read_text(encoding="utf-8"))
        self.gen = ra.generalization(self.d)
        self.emitted: set[str] = set()
        self.illustrative: set[str] = set()

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
            text = f"{value:.{digits}f}".replace("-", "−")
        return self._record(text, self.emitted)

    def illus(self, value, digits=2) -> str:
        text = f"{value:.{digits}f}"
        return self._record(text, self.illustrative)

    # lookups (P2 throughout) --------------------------------------------------
    def main(self, method, metric, dataset="mean", col="mean"):
        return ra.prot(self.d, dataset, method, metric, col)

    def pair(self, method, metric, reference="tda", dataset="mean"):
        return ra.paired(self.d, method, metric, reference, dataset)

    def comp(self, method, metric, dataset="mean"):
        return ra.comp(self.d, method, metric, dataset)

    def comp_pair(self, method, metric, dataset="mean"):
        return ra.comp_paired(self.d, method, metric, dataset)

    def val(self, column, dataset="mean"):
        t = self.d["val"]
        return float(t[t.dataset == dataset].iloc[0][column])

    def test(self, column, dataset="mean"):
        t = self.d["test"]
        return float(t[t.dataset == dataset].iloc[0][column])

    def rho(self, signal="conf", datasets="all"):
        t = self.d["rho"]
        return t[(t.signal == signal) & (t.datasets == datasets)].iloc[0]



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
              align=PP_ALIGN.RIGHT, bold_first_col=False, cell_colors=None, aligns=None, strong_rows=(),
              bold_cells=(), tint_rows=()):
    """Text columns can be left-aligned via `aligns` ("L"/"R" per column). `highlight_rows`: the gate (teal tint,
    bold); `strong_rows`: mean rows (neutral tint, bold); `bold_cells`: (row, col) pairs, e.g. the best per column."""
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
        strong = (r_index - 1) in strong_rows
        for col, text in enumerate(row):
            cell = table.cell(r_index, col)
            cell.text = ""
            cell.fill.solid()
            tinted = emphasised or (r_index - 1) in tint_rows
            cell.fill.fore_color.rgb = TINT_GATE if tinted else TINT_NEUTRAL if strong else SURFACE
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
            bold = emphasised or strong or (bold_first_col and col == 0) or (r_index - 1, col) in bold_cells
            _style(run, font_size, bold=bold, color=color)
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
    run.text = "Agreement-gated TDA  ·  Machlin & Yehuda"
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



# --- Slides -----------------------------------------------------------------------
# Each builder: (prs, r) -> slide. Notes are spoken English; numbers via r.num.

def slide_figure(name: str, draw, rc=None) -> Path:
    """Render a report figure again at slide size, with projector-sized text (`paper_figures.SLIDE_RC`)."""
    import matplotlib.pyplot as plt
    path = ASSETS / f"slide_{name}.png"
    ASSETS.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(rc or pf.SLIDE_RC):
        draw(path)
    return path


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
             [("TAU  ·  DEEP LEARNING  ·  FINAL PROJECT", 16, True, lavender)])
    add_text(slide, Inches(1.1), Inches(2.15), Inches(11.2), Inches(1.8),
             [("The cache that echoes CLIP: agreement-gated test-time adaptation", 40, True, light)])
    add_text(slide, Inches(1.1), Inches(4.25), Inches(11.2), Inches(1.6), [
        ("Gal Machlin   ·   Idan Yehuda", 22, False, light),
        ("Why TDA makes CLIP overconfident while raising its accuracy, where exactly it happens, "
         "and a fix that adds no parameters.", 20, False, soft),
    ], spacing=14)
    set_notes(slide, f"""
Hello everyone. This talk is about test-time adaptation of CLIP, and in particular about TDA, a
popular training-free method that improves CLIP's accuracy but makes its confidence less
trustworthy. We show where that comes from, prove the key step, and propose a fix that adds no
parameters: on our five design test sets it keeps TDA's accuracy and lowers its ECE from
{r.num(r.main('tda', 'ece'))} to {r.num(r.main('cal_tda_agree', 'ece'))}.
""")
    return slide


def slide_task(prs, r):
    slide = new_slide(prs, "The task: zero-shot CLIP on an unlabeled test stream", kicker="1 · Task")
    top1, top2, h = Inches(1.75), Inches(3.05), Inches(0.95)
    flow_box(slide, MARGIN, top1, Inches(2.3), h, "test image x", fill=SURFACE)
    arrow(slide, MARGIN + Inches(2.38), top1 + Inches(0.32), Inches(0.5), Inches(0.32))
    flow_box(slide, MARGIN + Inches(2.95), top1, Inches(3.3), h, "image encoder f", "frozen",
             fill=TINT_CLIP, border=CLIPBLUE)
    flow_box(slide, MARGIN, top2, Inches(2.3), h, "class name k", fill=SURFACE)
    arrow(slide, MARGIN + Inches(2.38), top2 + Inches(0.32), Inches(0.5), Inches(0.32))
    flow_box(slide, MARGIN + Inches(2.95), top2, Inches(3.3), h, "text encoder h", "frozen",
             fill=TINT_CLIP, border=CLIPBLUE)
    add_text(slide, MARGIN, top2 + h + Inches(0.08), Inches(6.3), Inches(0.4),
             [("prompt:  “a photo of a {class}.”", SUB, True, ACCENT)])
    arrow(slide, MARGIN + Inches(6.35), top1 + Inches(0.95), Inches(0.5), Inches(0.4))
    add_panel(slide, MARGIN + Inches(7.0), top1, Inches(5.1), Inches(2.6), fill=SURFACE)
    add_equation(slide, r"z_k = s\,\cos\left(f(x),\,h(t_k)\right)", MARGIN + Inches(7.1), top1 + Inches(0.2),
                 size=24, center_width=Inches(4.9))
    add_equation(slide, r"p = \mathrm{softmax}(z),\quad \hat{y}_0 = \arg\max_k\, z_k",
                 MARGIN + Inches(7.1), top1 + Inches(1.0), size=24, center_width=Inches(4.9))
    add_equation(slide, r"\mathrm{confidence} = \max_k\, p_k",
                 MARGIN + Inches(7.1), top1 + Inches(1.8), size=24, center_width=Inches(4.9))
    add_bullets(slide, [
        "Given: class names and unlabeled test images, one at a time. No labels, not even for tuning.",
        "Each prediction is final before the next image, and should come with an honest confidence.",
        f"Zero-shot CLIP is weak on unusual or fine-grained images: {r.num(r.main('clipzs', 'accuracy'))}% on DTD, "
        "Flowers, Pets, EuroSAT, Aircraft (our five design sets).",
    ], top=Inches(4.75), spacing=8)
    set_notes(slide, f"""
The task. We are given the class names, a pretrained CLIP, and a stream of unlabeled test images. CLIP
has an image encoder and a text encoder, both frozen; the logit of a class is the cosine between the
image feature and the feature of a prompt with the class name. Its argmax is the prediction, which we
call CLIP's pseudo-label, and its top probability is the confidence. There are no labels anywhere, each
prediction is final before the next image arrives, and we want both the right class and a believable
confidence. On textures, flowers, pet breeds, satellite images and aircraft models, our five design
datasets, zero-shot CLIP reaches only {r.num(r.main('clipzs', 'accuracy'))} percent on average. Test-time adaptation tries to fix that.
""")
    return slide


TDA_CITE = "Karmanov et al., CVPR'24 (TDA), arxiv.org/abs/2403.18293."


def slide_tda_why(prs, r):
    slide = new_slide(prs, "Two ways to adapt CLIP at test time", kicker="2 · TDA")
    add_picture(slide, TDA_FIGS / "Prompt-Tuning.png", MARGIN, Inches(1.6), Inches(6.2), Inches(2.3))
    add_picture(slide, TDA_FIGS / "TDA.png", MARGIN, Inches(3.95), Inches(6.2), Inches(2.4))
    add_bullets(slide, [
        "Episodic, with backprop (top: TPT; also C-TPT): per image, augmented views and gradient steps on the "
        "prompt, then reset.",
        "Online, training-free (bottom: TDA; also ECALP, and DMN without its augmented views): a memory of past "
        "test images, one forward pass per image.",
        "We study TDA (CVPR'24), a widely used online baseline.",
    ], left=Inches(7.1), top=Inches(1.7), width=Inches(5.6), spacing=12)
    add_text(slide, MARGIN, Inches(6.45), CONTENT_W, Inches(0.4), [("Fig. 1 of " + TDA_CITE, SMALL, False, MUTED)])
    set_notes(slide, """
Test-time adaptation uses the unlabeled stream to improve CLIP, and there are two families. This is the
first figure of the TDA paper. On top, test-time prompt tuning: for every image it builds many augmented
views and optimizes the prompt by backpropagation, then resets, which is slow. Below, TDA: no training,
only a cache of earlier test images labeled by CLIP, used for the next images. It is fast and fully
online, which is why it became a widely used baseline of the training-free family and why we study it.
""")
    return slide


def slide_tda_pipeline(prs, r):
    slide = new_slide(prs, "TDA's caches are labeled by CLIP's own predictions", kicker="2 · TDA")
    add_picture(slide, TDA_FIGS / "TDA-full.png", MARGIN, Inches(1.55), CONTENT_W, Inches(3.6))
    add_bullets(slide, [
        "Positive cache: per class, the most confident test images (lowest entropy); it votes for classes.",
        "Negative cache: uncertain images with the classes CLIP ruled out; it votes against them.",
        "Both store CLIP's own predictions as their labels.",
    ], top=Inches(5.25), spacing=6, size=SUB)
    add_text(slide, MARGIN, Inches(6.5), CONTENT_W, Inches(0.35), [("Fig. 2 of " + TDA_CITE, SMALL, False, MUTED)])
    set_notes(slide, """
This is TDA's pipeline, from its paper. CLIP produces a prediction. Two caches store test features as
keys and pseudo-labels as values: a positive cache of confident samples, and a negative cache of
uncertain samples that records which classes are unlikely. Both caches produce predictions that are
added to CLIP's. The detail that matters for us is the update path at the bottom: the cache labels are
CLIP's own predictions.
""")
    return slide


def slide_tda_math(prs, r):
    slide = new_slide(prs, "TDA in one equation", kicker="2 · TDA")
    a, b, s = r.num(r.cfg["alpha"], 0), r.num(r.cfg["beta"], 0), r.num(r.cfg["shot_capacity"], 0)
    add_panel(slide, MARGIN, Inches(1.65), CONTENT_W, Inches(2.0))
    add_equation(slide, r"o(x) = b(x) + \alpha\,c(x), \qquad b(x) = z(x) - n(x)",
                 MARGIN, Inches(1.85), size=26, center_width=CONTENT_W)
    add_equation(slide, r"c_k(x) = \sum_{i \in \mathcal{C}_k} e^{-\beta\,(1 - \langle f(x), f_i\rangle)} \in [0, S]",
                 MARGIN, Inches(2.6), size=26, center_width=CONTENT_W)
    add_bullets(slide, [
        "z: CLIP's logits; its argmax ŷ₀ is the label written to the cache.",
        "n: the negative cache's penalty; b is CLIP after the negative cache.",
        f"c: similarity-weighted votes of at most S = {s} cached images per class; each vote is at most 1.",
        "Each image is written to the cache (if it enters) before it is read.",
        f"One setting for all datasets, from TDA's paper: α = {a}, β = {b}.",
    ], top=Inches(3.9), spacing=10)
    set_notes(slide, f"""
Here is TDA in one line. The output is b, which is CLIP's logits after the negative cache's penalty, plus
alpha times the positive cache's votes. Each cached image votes with a kernel of its cosine to the test
image, so each vote is at most one, and with {s} images per class a class gets at most {s}. Two facts
matter later: the cache is filled with CLIP's argmax, and each image is written before it is read. We use
TDA's own single setting, alpha {a} and beta {b}, on every dataset, since an online method should not need
per-dataset tuning.
""")
    return slide


def render_calibration_figure(out_path: Path) -> Path:
    """Textbook reliability diagram from SYNTHETIC curves (concept only)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    confidence = np.linspace(0.30, 0.975, 14)
    over, under = confidence ** 2.4, confidence ** 0.45
    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#4D4D4D", linewidth=1.8, label="calibrated")
    ax.plot(confidence, over, marker="o", markersize=6, color="#B34A3C", linewidth=2.2, label="overconfident")
    ax.plot(confidence, under, marker="s", markersize=5, color="#4C72B0", linewidth=2.2, label="underconfident")
    pick = 10
    ax.plot([confidence[pick]] * 2, [over[pick], confidence[pick]], color="#7D5BA6", linewidth=3.2,
            label="one bin's gap")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("confidence (bin mean)", fontsize=14)
    ax.set_ylabel("accuracy in the bin", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=12, frameon=True, framealpha=0.95, edgecolor="#D8D5D0")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def slide_calibration(prs, r):
    slide = new_slide(prs, "Calibration: confidence should equal accuracy", kicker="2 · TDA's catch")
    add_panel(slide, MARGIN, Inches(1.65), Inches(7.0), Inches(1.4))
    add_equation(slide, r"\mathrm{ECE} = \sum_{m} \frac{|B_m|}{N}\,\left|\,\mathrm{acc}(B_m) - \mathrm{conf}(B_m)\,\right|",
                 MARGIN, Inches(1.85), size=26, center_width=Inches(7.0))
    add_bullets(slide, [
        "Bin predictions by confidence; ECE averages |accuracy − confidence| over the bins. Lower is better.",
        "The confidence − accuracy gap has a sign: above 0 overconfident, below 0 underconfident.",
        "When confidence is acted on, overconfidence is worse: the model acts on wrong predictions it trusts.",
    ], top=Inches(3.3), width=Inches(7.0), spacing=12)
    path = render_calibration_figure(ASSETS / "what_is_calibration.png")
    add_picture(slide, path, Inches(8.0), Inches(1.6), Inches(4.7), Inches(4.4))
    add_text(slide, Inches(8.0), Inches(6.05), Inches(4.7), Inches(0.4), [
        ("Concept sketch with synthetic curves, not data.", SMALL, False, MUTED)], align=PP_ALIGN.CENTER)
    set_notes(slide, """
Calibration means that the confidence a model reports matches how often it is right. The expected
calibration error bins the predictions by confidence and averages, over the bins, the gap between mean
confidence and accuracy, weighted by the number of samples in each bin. The sketch on the right is a
concept drawing, not data: below the diagonal is overconfident, above it underconfident. ECE has no
sign, so we also report the signed gap, confidence minus accuracy. When the confidence is used to decide
whether to trust a prediction, overconfidence is the worse error: the model acts on wrong predictions it
trusts, while an underconfident one only gives up some predictions it could have kept.
""")
    return slide


def slide_tda_overconfident(prs, r):
    slide = new_slide(prs, "TDA's catch: more accuracy, worse calibration", kicker="2 · TDA's catch")
    rows = []
    for ds in CORE + ("mean",):
        rows.append([NAMES.get(ds, "Mean"), r.num(r.main("clipzs", "accuracy", ds)), r.num(r.main("tda", "accuracy", ds)),
                     r.num(r.main("clipzs", "ece", ds)), r.num(r.main("tda", "ece", ds))])
    add_table(slide, ["dataset (test)", "CLIP acc.", "TDA acc.", "CLIP ECE", "TDA ECE"], rows, MARGIN, Inches(1.7),
              Inches(7.4), col_widths=[2.0, 1.2, 1.2, 1.2, 1.2], row_height=Inches(0.48), font_size=17,
              strong_rows=(5,))
    a = r.d["ni_abl"].set_index("tier").loc["test"]
    up = sum(r.main("tda", "ece", ds) > r.main("clipzs", "ece", ds) for ds in CORE)
    add_bullets(slide, [
        f"Accuracy up on every dataset; ECE up on {r.num(up, 0)} of {r.num(len(CORE), 0)}, sharply on DTD, Flowers "
        "and Aircraft.",
        f"Which cache? Mean ECE over {r.num(r.d['ni_abl_seed'].seed.nunique(), 0)} orders: CLIP {r.num(a.clip_ece, 1)} "
        f"→ with the negative cache {r.num(a.base_ece, 1)} → with both caches (TDA) {r.num(a.tda_ece, 1)}.",
        "The positive cache brings the overconfidence. Why?",
    ], left=Inches(8.4), top=Inches(1.8), width=Inches(4.3), spacing=14, size=SUB)
    seeds = r.num(r.pair("cal_tda_agree", "ece").n, 0)
    set_notes(slide, f"""
Per dataset, on the test splits with {seeds} stream orders. TDA gains accuracy everywhere, but its ECE grows
on three of the five datasets, sharply on DTD, from {r.num(r.main('clipzs', 'ece', 'dtd'))} to {r.num(r.main('tda', 'ece', 'dtd'))}. Pets and EuroSAT are the
exceptions; on Pets CLIP is underconfident, and TDA's extra confidence happens to correct it, so keep Pets
in mind. Which of the two caches is responsible? The negative cache alone lowers ECE; adding the positive
cache takes it to {r.num(a.tda_ece, 1)} over three orders. So the problem sits in the positive cache. Why?
""")
    return slide


# --- Diagnosis and method --------------------------------------------------------------
def slide_selfvote(prs, r):
    slide = new_slide(prs, "Symptom: each new cache entry votes for itself", kicker="3 · Diagnosis · val")
    v = r.d["val"][r.d["val"].dataset != "mean"]
    lo, hi = r.num(v.self_vote_share.min(), 0), r.num(v.self_vote_share.max(), 0)
    path = slide_figure("selfvote", lambda p: pf.fig_selfvote(r.d["val"], r.d["read_first"], p, figsize=(12.0, 3.4),
                                                              letters=False))
    add_picture(slide, path, MARGIN, Inches(1.5), CONTENT_W, Inches(3.5))
    add_bullets(slide, [
        "Written before read: a new cache entry votes for itself with weight exp(0) = 1.",
        f"Frequent: {lo}–{hi}% of samples. Self-voters agree with CLIP {r.num(v.p_agree_given_self.min(), 0)}–"
        f"{r.num(v.p_agree_given_self.max(), 0)}% of the time, the others {r.num(v.p_agree_given_no_self.min(), 0)}–"
        f"{r.num(v.p_agree_given_no_self.max(), 0)}% (left).",
        f"Read-first TDA: ECE {r.num(r.val('tda_ece'))} → {r.num(ra.read_first(r.d, 'val', 'ece'))}, still above "
        f"CLIP's {r.num(r.val('clip_ece'))} (right).",
    ], top=Inches(5.1), spacing=6, size=SUB)
    set_notes(slide, f"""
The first symptom. Because each image is written before it is read, an image that enters the cache votes
for its own pseudo-label with weight exactly one. That vote adds nothing to CLIP's own prediction. It
happens for {lo} to {hi} percent of the samples on the validation splits, and self-voters agree with CLIP
far more often than the others, left panel. Reading the cache before writing removes the vote and lowers
TDA's ECE, right panel, but not to CLIP's level. So the self-vote matters, but it is not the whole story.
""")
    return slide


def slide_lemma(prs, r):
    slide = new_slide(prs, "The echo: where the cache agrees, it only adds confidence", kicker="3 · Diagnosis · val")
    add_panel(slide, MARGIN, Inches(1.5), CONTENT_W, Inches(1.85), fill=TINT_ACCENT, line=ACCENT)
    add_text(slide, MARGIN + Inches(0.25), Inches(1.6), Inches(11.6), Inches(0.4),
             [("Lemma (agreement). If argmax b = argmax c = ŷ, then for every j ≠ ŷ", SUB, True, INK)])
    add_equation(slide, r"o_{\hat y} - o_j = (b_{\hat y} - b_j) + \alpha\,(c_{\hat y} - c_j) \;\geq\; b_{\hat y} - b_j > 0",
                 MARGIN, Inches(2.05), size=26, center_width=CONTENT_W)
    add_text(slide, MARGIN + Inches(0.25), Inches(2.85), Inches(11.6), Inches(0.45),
             [("so the decision is unchanged, no margin shrinks, and max softmax(o) ≥ max softmax(b).", SUB, False, INK)])
    path = slide_figure("agreement", lambda p: pf.fig_agreement(r.d["val"], p, figsize=(7.4, 3.5), letters=False))
    add_picture(slide, path, MARGIN, Inches(3.45), Inches(7.4), Inches(3.4), align="left")
    share = r.num(r.val("agree_share"), 0)
    add_bullets(slide, [
        f"{share}% of samples agree: accuracy unchanged, gap {r.num(r.val('base_gap_agree'), 1, sign=True)} → "
        f"{r.num(r.val('tda_gap_agree'), 1, sign=True)} points (red).",
        "All of the accuracy gain is on the disagreeing samples (teal).",
        "Agreement is the common case: the cache echoes CLIP.",
    ], left=Inches(8.3), top=Inches(3.65), width=Inches(4.45), spacing=10, size=SUB)
    set_notes(slide, f"""
Now the general mechanism, with a short proof. Let b be CLIP after the negative cache. If the positive
cache's top class is the same as b's, every margin of the output is b's margin plus alpha times a
non-negative cache margin: the decision cannot change, and the top probability cannot drop. The data
match. On the validation splits {share} percent of the samples agree; there the positive cache leaves accuracy
unchanged but turns a slight underconfidence into strong overconfidence. Its whole accuracy gain comes from
the samples where it disagrees. Since the cache is filled with CLIP's own labels, agreement is the common
case: the cache echoes CLIP.
""")
    return slide


def slide_method(prs, r):
    slide = new_slide(prs, "The agreement gate: use the cache only where it disagrees", kicker="3 · Method")
    add_panel(slide, MARGIN, Inches(1.65), CONTENT_W, Inches(1.4), fill=TINT_ACCENT, line=ACCENT)
    add_equation(slide, r"\tilde o(x) = b(x) + \alpha\,g(x)\,c(x), \qquad g(x) = \mathbf{1}\left[\arg\max_k c_k(x) \neq \hat y_0(x)\right]",
                 MARGIN, Inches(1.95), size=28, center_width=CONTENT_W)
    held = r.gen[r.gen.tier.isin(("near", "far"))]
    add_bullets(slide, [
        "Caches, admission, negative cache, α, β: TDA's, unchanged. No new parameter, no label, one comparison "
        "per image.",
        "Compares with ŷ₀, the label the cache is filled with, so agreement with it is the echo itself "
        f"(ŷ₀ differs from b's top class on only {r.num(r.val('agree_neg_moved'), 1)}% of agreeing samples).",
        "Designed on the val splits of the five design sets, then frozen; tested on their test splits and on "
        f"{r.num(len(held), 0)} held-out sets.",
        "Precedent: update only on disagreement (Malach & Shalev-Shwartz, NeurIPS'17), here at inference time.",
    ], top=Inches(3.35), spacing=12)
    set_notes(slide, f"""
The method follows directly. We keep everything in TDA and switch the positive cache off on samples where
its top class agrees with CLIP's pseudo-label, because there it can only inflate confidence. Where it
disagrees, the cache is used exactly as in TDA. Why CLIP's label and not b's top class, as in the lemma?
Because CLIP's label is what the cache is filled with, so agreement with it is the echo; the two differ on
only {r.num(r.val('agree_neg_moved'), 1)} percent of agreeing samples. The rule adds no parameter and needs no label, and it was fixed on the
validation splits before any of its own test or held-out results were computed.
""")
    return slide


# --- Results ---------------------------------------------------------------------------------
def slide_main(prs, r):
    slide = new_slide(prs, "Main result: TDA's accuracy at CLIP's calibration", kicker="4 · Results · test")
    methods = ("clipzs", "tda", "cal_tda_agree")
    rows, bold = [], set()
    for i, ds in enumerate(CORE + ("mean",)):
        rows.append([NAMES.get(ds, "Mean")] + [r.num(r.main(m, metric, ds)) for metric in ("accuracy", "ece")
                                               for m in methods])
        for j, metric in enumerate(("accuracy", "ece")):
            vals = [round(r.main(m, metric, ds), 2) for m in methods]
            best = max(vals) if metric == "accuracy" else min(vals)
            bold |= {(i, 1 + 3 * j + k) for k, v in enumerate(vals) if v == best}
    add_table(slide, ["dataset", "CLIP acc.", "TDA acc.", "gate acc.", "CLIP ECE", "TDA ECE", "gate ECE"], rows, MARGIN,
              Inches(1.75), Inches(8.2), col_widths=[1.6, 1, 1, 1, 1, 1, 1], row_height=Inches(0.46), font_size=16,
              bold_cells=bold)
    add_text(slide, MARGIN, Inches(5.1), Inches(8.2), Inches(0.4),
             [("Bold: best of the three methods per dataset, for accuracy and for ECE.", SMALL, False, MUTED)])
    pa, pe = r.pair("cal_tda_agree", "accuracy"), r.pair("cal_tda_agree", "ece")
    n = r.num(pe.n, 0)
    level = next(int(c[2:-3]) for c in pa.index if c.startswith("ci") and c.endswith("_lo"))  # from "ci90_lo"
    gain = r.main("tda", "accuracy") - r.main("clipzs", "accuracy")
    add_text(slide, Inches(9.1), Inches(1.8), Inches(3.6), Inches(4.6), [
        (f"Gate − TDA, {n} stream orders [{r.num(level, 0)}% CI]", SUB, True, ACCENT),
        (f"accuracy {r.num(pa.mean_diff, sign=True)} [{r.num(pa.ci90_lo, sign=True)}, {r.num(pa.ci90_hi, sign=True)}]",
         SUB, False, INK),
        (f"(TDA vs CLIP: {r.num(gain, sign=True)})", SUB, False, MUTED),
        (f"ECE {r.num(pe.mean_diff, sign=True)} [{r.num(pe.ci90_lo, sign=True)}, {r.num(pe.ci90_hi, sign=True)}]",
         SUB, True, INK),
        (f"Mean ECE lower on all {n} orders; per dataset, higher only on Pets (CLIP underconfident).", SUB, False, INK),
    ], spacing=10)
    set_notes(slide, f"""
The main result on the five design test sets, {n} stream orders. The gate costs a small but consistent
{r.num(-pa.mean_diff)} accuracy points, small next to TDA's gain over CLIP, and it lowers the ECE on every order, to
{r.num(r.main('cal_tda_agree', 'ece'))}, just below CLIP. Per dataset it is far better calibrated than TDA on DTD, Flowers,
EuroSAT and Aircraft, and worse on Pets, where CLIP is underconfident.
""")
    return slide


COMPARISON = (("CLIP", "CLIP"), ("TDA", "TDA"), ("agreement gate", "agreement gate"),
              ("read-first TDA", "read-first TDA"), ("TDA + T (val)", "TDA + temperature (fit on val)"),
              ("TDA + ZAEC", "TDA + ZAEC"), ("gate + ZAEC", "gate + ZAEC"),
              ("DMN-weak", "DMN-weak"), ("ECALP", "ECALP"))


def slide_comparison(prs, r):
    slide = new_slide(prs, "Lower mean ECE than post-hoc calibration, and the two combine", kicker="4 · Results · test")
    rows = []
    for method, label in COMPARISON[1:]:
        acc, ece = r.comp(method, "accuracy"), r.comp(method, "ece")
        delta = "—" if method == "TDA" else r.num(r.comp_pair(method, "ece").mean_diff, sign=True)
        rows.append([label, r.num(acc), r.num(ece), delta])
    add_table(slide, ["method (mean of 5)", "acc.", "ECE", "ΔECE vs TDA"], rows, MARGIN, Inches(1.7),
              Inches(7.6), col_widths=[3.6, 1, 1, 1.5], row_height=Inches(0.47), font_size=16, highlight_rows=(1,))
    tda, clip, rf = r.comp("TDA", "ece"), r.comp("CLIP", "ece"), r.comp("read-first TDA", "ece")
    add_bullets(slide, [
        f"Read-first TDA removes {r.num(100 * (tda - rf) / (tda - clip), 0)}% of TDA's ECE above CLIP's: the "
        "self-vote is only part of the echo.",
        "On the mean the gate beats a single temperature fitted with labels and the label-free ZAEC (the temperature "
        "wins on DTD and Aircraft, ZAEC on Flowers; B1).",
        "The gate and ZAEC act at different points: gate + ZAEC is the lowest.",
        f"DMN-weak and ECALP: higher ECE than TDA; ECALP is the most accurate ({r.num(r.comp('ECALP', 'accuracy'))}%).",
    ], left=Inches(8.6), top=Inches(1.75), width=Inches(4.1), spacing=12, size=SUB)
    add_text(slide, MARGIN, Inches(6.1), Inches(7.6), Inches(0.4), [
        (f"{r.num(r.comp_pair('agreement gate', 'ece').n, 0)} stream orders; TDA's ECE reads {r.num(tda)} here and "
         f"{r.num(r.main('tda', 'ece'))} on the main-result slide.", SMALL, False, MUTED)])
    set_notes(slide, f"""
Here we compare with the closest alternatives, paired to TDA on the same stream orders. A single
temperature, shared by the five datasets and fitted with labels on their validation splits, reaches an ECE
of {r.num(r.comp('TDA + T (val)', 'ece'))}; ZAEC, a recent label-free post-hoc method that restores the zero-shot entropy, reaches
{r.num(r.comp('TDA + ZAEC', 'ece'))}. The gate reaches {r.num(r.comp('agreement gate', 'ece'))} without any calibration step, because it removes the cause instead of
rescaling afterwards, and the two combine. DMN-weak and ECALP, run like TDA with one view per image, have
higher ECE than TDA.
""")
    return slide


def slide_heldout(prs, r):
    g = r.gen[r.gen.tier.isin(("near", "far"))]
    far = g[g.tier == "far"]
    slide = new_slide(prs, f"Held-out: same accuracy; lower ECE on {r.num((far.d_ece < 0).sum(), 0)} of "
                           f"{r.num(len(far), 0)} far sets", kicker="4 · Results · held-out")
    rows = []
    for tier in ("near", "far"):
        m = g[g.tier == tier].mean(numeric_only=True)
        rows.append([f"{tier} ({r.num((g.tier == tier).sum(), 0)} sets)"]
                    + [r.num(m[f"{k}_{metric}"]) for metric in ("accuracy", "ece") for k in ("clip", "tda", "gate")])
    add_table(slide, ["tier (mean)", "CLIP acc.", "TDA acc.", "gate acc.", "CLIP ECE", "TDA ECE", "gate ECE"], rows,
              MARGIN, Inches(1.7), Inches(9.4), col_widths=[2.0, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1], row_height=Inches(0.5),
              font_size=17)
    big, rest = ra.far_spread(g)
    add_bullets(slide, [
        "Near (" + ", ".join(NAMES.get(x, x) for x in g[g.tier == "near"].dataset)
        + "): ordinary photos; CLIP is already accurate and calibrated, and little changes.",
        "Far: medical scans, digits, rendered text, sketches and renditions, facial expressions, traffic signs. "
        "TDA adapts, and adds ECE.",
        f"ECE vs TDA on far: {r.num(big.d_ece.min(), 1, sign=True)} to {r.num(big.d_ece.max(), 1, sign=True)} on "
        + ", ".join(NAMES.get(x, x) for x in big.dataset) + f"; {r.num(rest.d_ece.min(), 1, sign=True)} to "
        f"{r.num(rest.d_ece.max(), 1, sign=True)} on the other {r.num(len(rest), 0)}.",
        "TDA stays better calibrated on ImageNet-R (CLIP underconfident) and GTSRB.",
    ], top=Inches(3.45), spacing=10, size=SUB)
    set_notes(slide, f"""
Now data that played no part in the design, in two tiers defined by what the images are. Near sets are
ordinary photos like CLIP's training data: CLIP is already accurate and calibrated, and neither method
changes much. Far sets are medical scans, digits, rendered text, sketches and renditions, facial expressions
and traffic signs: TDA adds accuracy there, and ECE. The gate keeps TDA's accuracy on every held-out set, and on far its
mean ECE, {r.num(far.gate_ece.mean())}, ends below CLIP's {r.num(far.clip_ece.mean())}. The gain is uneven: large on PCam, ImageNet-Sketch and MNIST,
small on the others, and TDA stays better calibrated on ImageNet-R and GTSRB.
""")
    return slide


ABL_TIERS = (("test", "design (test)"), ("near", "near"), ("far", "far (no ImageNet-Sketch)"))
ABL_ROWS = (("clip", "CLIP"), ("base", "+ negative cache = b"), ("apos", "b + positive cache on agreeing samples"),
            ("gate", "b + positive cache on disagreeing samples (gate)"),
            ("rand", "b + positive cache on as many random samples"), ("tda", "b + positive cache on all samples (TDA)"))


def slide_ablation(prs, r):
    slide = new_slide(prs, "Ablation: agreement picks the right samples", kicker="4 · Results · all tiers")
    a = r.d["ni_abl"].set_index("tier")
    rows = [[label] + [f"{r.num(a.loc[t, f'{k}_acc'], 1)} / {r.num(a.loc[t, f'{k}_ece'], 1)}" for t, _ in ABL_TIERS]
            for k, label in ABL_ROWS]
    rows.append(["samples the gate opens on"] + [f"{r.num(a.loc[t, 'disagree_share'], 0)}%" for t, _ in ABL_TIERS])
    add_table(slide, ["accuracy / ECE (3 stream orders)"] + [lab for _, lab in ABL_TIERS], rows, MARGIN, Inches(1.55),
              CONTENT_W, col_widths=[4.6, 2.1, 2.1, 2.6], row_height=Inches(0.42), font_size=15, highlight_rows=(3,))
    held = a.loc[[t for t, _ in ABL_TIERS]]
    d_apos = (held.apos_acc - held.base_acc).abs().max()
    sw = ra.sweep_values(r.d["matched"], r.d["alpha"])
    add_bullets(slide, [
        f"Agreeing samples: accuracy moves ≤ {r.num(d_apos)} points, ECE doubles on design. Only confidence.",
        "Not just less cache: as many random samples are worse in accuracy and ECE on design and far; a smaller α "
        f"matches the gate's accuracy only at α ≈ {r.num(sw['alpha'], 1)}, with ECE {r.num(sw['ece'], 1)}.",
        "Near: CLIP is underconfident, so extra confidence happens to lower ECE.",
    ], top=Inches(5.0), spacing=8, size=SUB)
    set_notes(slide, f"""
The ablation tests the diagnosis on the design test sets and both held-out tiers; the far column leaves out
ImageNet-Sketch, of which we logged only the first samples, so it differs a little from the previous slide.
Adding TDA's positive cache only on the agreeing samples barely moves accuracy but doubles the ECE on the
design sets: only confidence. Is the gate just using less cache? No. Adding the cache on as many random
samples is worse in both accuracy and ECE on the design and far sets, {r.num(a.loc['test', 'rand_ece'], 1)} against the gate's {r.num(a.loc['test', 'gate_ece'], 1)} on design, and turning down
TDA's alpha until it matches the gate's accuracy leaves its ECE at {r.num(sw['ece'], 1)}. Which samples matters.
""")
    return slide


def slide_where(prs, r):
    slide = new_slide(prs, "When does it help? Mostly where CLIP is not underconfident", kicker="5 · Discussion · all tiers")
    rg, rc = r.rho("clip_gap"), r.rho("conf")
    path = slide_figure("where", lambda p: pf.fig_where(r.d["helps"], r.d["rho"], p, figsize=(6.6, 4.9),
                                                       offset_scale=1.5, label_size="x-small"))
    add_picture(slide, path, MARGIN, Inches(1.5), Inches(6.6), Inches(5.0))
    gaps = r.d["gaps"]
    g = gaps[gaps.tier.isin(("test", "near", "far"))]
    add_bullets(slide, [
        f"The more overconfident CLIP is, the more the gate lowers ECE: ρ = {r.num(rg.rho, sign=True)} over "
        f"{r.num(rg.n, 0)} sets. That gap needs labels; CLIP's mean confidence on the first "
        f"{r.num(SIGNAL_HEAD, 0)} images is related only weakly (ρ = {r.num(rc.rho, sign=True)}; "
        f"{r.num(r.rho('conf', 'far').rho, sign=True)} within far).",
        f"TDA raises CLIP's gap on {r.num((g.tda_gap > g.clip_gap).sum(), 0)} of {r.num(len(g), 0)} sets; the gate "
        f"lowers TDA's gap on all {r.num((g.gate_gap < g.tda_gap).sum(), 0)}.",
        "Where TDA's ECE is lower (named sets above zero), the gate is underconfident or calibrated in the mean: "
        "the safer side.",
    ], left=Inches(7.45), top=Inches(1.7), width=Inches(5.25), spacing=12, size=SUB)
    set_notes(slide, f"""
When does the gate help? On agreeing samples the echo only adds confidence, so removing it should help where
CLIP is calibrated or overconfident, and hurt where CLIP is underconfident, where the added confidence happens
to correct it. That is the tendency we see: the more overconfident CLIP is, the more the gate lowers ECE, with
a rank correlation of {r.num(rg.rho, sign=True)}. That gap needs labels, so it explains the result but cannot drive a switch. The
signed gap also shows the direction of the errors: TDA pushes toward overconfidence almost everywhere, and on
the sets where TDA wins in ECE, Pets, Caltech101, Food101, ImageNet-R and GTSRB, the gate is underconfident or
calibrated in the mean, the safer side.
""")
    return slide


def slide_flips(prs, r):
    slide = new_slide(prs, "No score of CLIP or the cache tells TDA's right flips from wrong ones",
                      kicker="5 · Discussion · held-out")
    path = slide_figure("flips", lambda p: ni.fig_deciles(r.d["ni_deciles"], r.d["ni_corr"], ni.DISCUSSED, p, ncols=3,
                                                         panel=(4.1, 3.4)))
    add_picture(slide, path, MARGIN, Inches(1.45), CONTENT_W, Inches(3.5))
    corr, dec, flt = r.d["ni_corr"], r.d["ni_deciles"], r.d["ni_filter"].set_index("tier")
    c = corr[corr.measure.isin(ni.DISCUSSED) & corr.tier.isin(("near", "far"))]
    d = dec[dec.measure.isin(ni.DISCUSSED) & dec.tier.isin(("near", "far"))]
    orc = flt.loc[["test", "near", "far"]].oracle_block
    add_bullets(slide, [
        f"Flip: TDA and CLIP disagree and one of them is right. Reverting exactly the wrong flips would add "
        f"{r.num(orc.min(), 1)}–{r.num(orc.max(), 1)} accuracy points.",
        f"Yet |ρ| ≤ {r.num(c.rho.abs().max())}, and about half or more of the flips are right in every decile "
        f"(lowest {r.num(d.right_pct.min(), 0)}%): no range is worth reverting.",
        "Choosing which flips to keep likely needs another source of information (augmented views, a second model).",
    ], top=Inches(5.05), spacing=8, size=SUB)
    set_notes(slide, f"""
The gate reads one bit, agreement. Could a finer use of the scores do better? TDA's accuracy gain comes from
flips, samples where its decision differs from CLIP's and one of the two is right. An oracle that reverted
exactly the wrong flips would add up to {r.num(orc.max(), 1)} points, so there is room. We test one score of each kind on
the held-out flips: CLIP's confidence, the cache's confidence, and their similarity. Their correlation with a
flip being right is at most {r.num(c.rho.abs().max())} in size, and for the cache's confidence it even changes sign between near and
far. In every decile about half or more of the flips are right, so no range is worth reverting. Seven more
scores and a regression on all ten do no better.
""")
    return slide


def slide_conclusion(prs, r):
    slide = new_slide(prs, "TDA's cache echoes CLIP; gating it on disagreement removes the echo", kicker="5 · Conclusion")
    g = r.gen[r.gen.tier == "far"]
    add_text(slide, MARGIN, Inches(1.75), CONTENT_W, Inches(3.4), [
        ("Where the cache agrees with CLIP, it cannot change the decision; it only adds confidence.", BODY, False, INK),
        (f"The gate keeps TDA's accuracy at CLIP-level ECE on the design sets ({r.num(r.main('tda', 'ece'))} → "
         f"{r.num(r.main('cal_tda_agree', 'ece'))}) and lowers ECE on {r.num((g.d_ece < 0).sum(), 0)} of "
         f"{r.num(len(g), 0)} far sets. No new parameter, no label.", BODY, False, INK),
        ("Beyond agreement, no score of CLIP or the cache tells right flips from wrong.", BODY, False, INK),
    ], spacing=18)
    gaps, helps = r.d["gaps"], r.d["helps"]
    sets = gaps[gaps.tier.isin(("test", "near", "far"))].merge(helps[["dataset", "d_ece"]], on="dataset")
    add_text(slide, MARGIN, Inches(4.9), CONTENT_W, Inches(1.8), [
        (f"Limits: worse than TDA in ECE on {r.num((sets.d_ece > 0).sum(), 0)} of {r.num(len(sets), 0)} sets, where "
         "CLIP is underconfident (Pets, Caltech101, Food101, ImageNet-R) and on GTSRB, always underconfident or "
         "calibrated in the mean; open-gate outputs stay overconfident on hard sets (DTD; ZAEC helps); one backbone "
         "and one cache method, though the argument should carry to any cache of the model's own pseudo-labels.",
         SUB, False, MUTED)])
    set_notes(slide, """
To close. TDA's positive cache mostly repeats CLIP's own decision, and where it does, it cannot change the
prediction, only inflate the confidence. Gating the cache on disagreement removes that echo: it keeps TDA's
accuracy and brings the calibration back to CLIP's level on the design sets, and on far data it removes much
of TDA's calibration cost. Beyond that one bit, the scores of CLIP and the cache do not say which corrections
are right. The gate is worse calibrated than TDA where CLIP is underconfident, though on the safer side, and
telling such data apart without labels is the open problem.
""")
    return slide


# --- Backup ---------------------------------------------------------------------------------
def slide_b_per_dataset(prs, r):
    slide = new_slide(prs, "Backup: comparison per dataset", kicker="Backup · Results · test")
    rows, bold = [], set()
    for i, (method, label) in enumerate(COMPARISON):
        rows.append([label] + [r.num(r.comp(method, "ece", ds)) for ds in CORE])
    for j, ds in enumerate(CORE):
        vals = [round(r.comp(m, "ece", ds), 2) for m, _ in COMPARISON]
        bold |= {(i, 1 + j) for i, v in enumerate(vals) if v == min(vals)}
    add_table(slide, ["ECE (3 stream orders)"] + [NAMES[d] for d in CORE], rows, MARGIN, Inches(1.7), CONTENT_W,
              col_widths=[3.4, 1, 1, 1, 1, 1], row_height=Inches(0.4), font_size=15, tint_rows=(2,),
              bold_cells=bold)
    add_text(slide, MARGIN, Inches(6.2), CONTENT_W, Inches(0.4), [("Bold: lowest ECE per dataset.", SMALL, False, MUTED)])
    set_notes(slide, """
Per-dataset ECE for every method in the comparison, three stream orders each. The gate is best on EuroSAT;
the temperature fitted on the validation splits wins on DTD and Aircraft, TDA plus ZAEC on Flowers, and
plain TDA on Pets, where removing the echo costs calibration. Combining the gate with ZAEC is the strongest
on the mean.
""")
    return slide


def slide_b_reproduction(prs, r):
    slide = new_slide(prs, "Backup: reproduction and TDA's per-dataset setting", kicker="Backup")
    rows = []
    for m, label in (("clipzs", "CLIP"), ("tda", "TDA"), ("cal_tda_agree", "agreement gate")):
        rows.append([label] + [r.num(ra.p1_mean(r.d, m, metric, col)) for metric in ("accuracy", "ece")
                               for col in ("mean_P1", "mean_P2")])
    add_table(slide, ["5 design test sets, 3 orders, direct replay", "acc. (per-set α, β)", "acc. (single)", "ECE (per-set)",
                      "ECE (single)"], rows, MARGIN, Inches(1.8), Inches(11.0), col_widths=[3.0, 2.0, 1.6, 1.8, 1.6],
              row_height=Inches(0.5), font_size=17)
    repro = r.d["repro"].delta_gpu_vs_benchmark.abs().max()
    diff = max(abs(ra.p1_mean(r.d, m, metric, "mean_P2") - r.comp(cm, metric))
               for m, cm in (("tda", "TDA"), ("cal_tda_agree", "agreement gate")) for metric in ("accuracy", "ece"))
    add_bullets(slide, [
        f"Our GPU runs reproduce the benchmark's CLIP and TDA accuracies within {r.num(repro)} points per dataset.",
        "TDA's code release ships per-dataset α, β; they do not do better than the paper's single setting.",
        f"This table replays TDA's own code; the comparison, ablation and discussion slides re-evaluate logged runs "
        f"of an equivalent implementation (within {r.num(diff)} points).",
    ], top=Inches(4.0), spacing=10, size=SUB)
    set_notes(slide, f"""
For completeness, the per-dataset alpha and beta that TDA's code release ships, against the single setting from
TDA's paper that we use throughout. The per-dataset values do not do better, and the conclusions do not change.
Our GPU reproduction of the benchmark is within {r.num(repro)} points per dataset; the analyses re-evaluate logged runs
of an equivalent implementation, which agree with this direct replay within {r.num(diff)} points on these means.
""")
    return slide


def slide_b_scores(prs, r):
    slide = new_slide(prs, "Backup: all ten label-free scores", kicker="Backup · held-out")
    small = {**pf.SLIDE_RC, "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11, "legend.fontsize": 10,
             "xtick.labelsize": 10, "ytick.labelsize": 10}
    path = slide_figure("scores", lambda p: ni.fig_deciles(r.d["ni_deciles"], r.d["ni_corr"], list(ni.MEASURES), p,
                                                          ncols=5, panel=(2.45, 2.3), wrap_titles=True), rc=small)
    add_picture(slide, path, MARGIN, Inches(1.45), CONTENT_W, Inches(4.5))
    rules, flt = r.d["ni_rules"], r.d["ni_filter"].set_index("tier")
    best = {t: rules[rules.tier == t].gain.max() for t in ("near", "far")}
    add_bullets(slide, [
        f"Best one-threshold rule per score, chosen on validation flips: {r.num(best['near'], sign=True)} accuracy "
        f"points on near, {r.num(best['far'], sign=True)} on far.",
        f"All ten in a logistic regression: {r.num(flt.loc['far', 'filter_gain'], sign=True)} points on far.",
    ], top=Inches(6.0), spacing=6, size=SUB)
    set_notes(slide, f"""
The same view for all ten label-free scores: three of CLIP, three of the cache and four of their relation. None
separates right from wrong flips on the held-out tiers. The best threshold rule per score, chosen on the
validation flips, gains nothing on the held-out tiers, {r.num(best['near'], sign=True)} points on near and {r.num(best['far'], sign=True)} on far, and all ten combined in a logistic
regression change accuracy by {r.num(flt.loc['far', 'filter_gain'], sign=True)} points on the far tier.
""")
    return slide


MAIN_BUILDERS = [
    slide_title, slide_task,
    slide_tda_why, slide_tda_pipeline, slide_tda_math, slide_calibration, slide_tda_overconfident,
    slide_selfvote, slide_lemma, slide_method,
    slide_main, slide_comparison, slide_heldout, slide_ablation,
    slide_where, slide_flips,
    slide_conclusion,
]
BACKUP_BUILDERS = [slide_b_per_dataset, slide_b_reproduction, slide_b_scores]
SLIDE_BUILDERS = MAIN_BUILDERS + BACKUP_BUILDERS
TDA_SECTION = [slide_tda_why, slide_tda_pipeline, slide_tda_math, slide_calibration, slide_tda_overconfident]


def slide_heading(slide, index):
    if index == 1:
        return "Title"
    for shape in slide.shapes:
        if shape.name == "title":
            return shape.text_frame.text.strip()
    return f"Slide {index}"


def write_markdown_mirror(prs, md_path) -> Path:
    lines = [
        "# The cache that echoes CLIP: agreement-gated TDA — deck v6",
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
