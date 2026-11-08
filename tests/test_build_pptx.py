"""The deck is built from tracked CSVs and records; these tests pin that."""

import json
import re
from pathlib import Path

import pandas as pd
import pytest
from pptx import Presentation
from pptx.util import Pt

from presentation.build_pptx import (
    BACKUP_BUILDERS, MAIN_BUILDERS, SLIDE_BUILDERS, TDA_SECTION, build, number_tokens,
)

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_IDS = {f"A{i}" for i in range(1, 11)} | {"T06", "T06 timing", "T07", "T08"}


def _slide_text(slide, *, skip_footer=True):
    parts = []
    for shape in slide.shapes:
        if skip_footer and shape.name == "footer":
            continue
        if shape.has_text_frame:
            parts.append(shape.text_frame.text)
        if shape.has_table:
            for row in shape.table.rows:
                parts.extend(cell.text for cell in row.cells)
    return "\n".join(parts)


@pytest.fixture(scope="module")
def deck(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("deck")
    out, results = build(out_dir / "deck.pptx", markdown_path=out_dir / "slides.md")
    prs = Presentation(str(out))
    texts = [_slide_text(s) for s in prs.slides]
    notes = [s.notes_slide.notes_text_frame.text for s in prs.slides]
    return prs, texts, notes, (out_dir / "slides.md").read_text(encoding="utf-8"), results


def test_slide_count_and_every_slide_has_speaker_notes(deck):
    prs, _, notes, markdown, _ = deck
    assert len(prs.slides) == len(SLIDE_BUILDERS) == 42
    assert len(MAIN_BUILDERS) == 36 and len(BACKUP_BUILDERS) == 6
    for index, note in enumerate(notes, start=1):
        assert len(note.split()) >= 25, f"slide {index} has no real speaker notes"
    assert markdown.count("\n## ") == len(SLIDE_BUILDERS)
    for note in notes:   # the markdown mirrors the notes verbatim
        assert note.strip() in markdown


def test_number_scanner_finds_data_but_not_names():
    text = "ECE 17.41 and 1,692, rose −5.52 (p > 0.03) on ViT-B/16, CVPR'24, A3, top-2, 16-shot, 100 classes"
    assert number_tokens(text) == ["17.41", "1,692", "5.52", "0.03", "100"]


def test_tda_section_is_at_least_a_quarter_of_the_main_deck():
    assert len(TDA_SECTION) / len(MAIN_BUILDERS) >= 0.25


def test_slides_name_no_code_identifiers(deck):
    _, texts, _, _, _ = deck
    for index, text in enumerate(texts, start=1):
        for token in ("cal_tda", "clipzs", "fusion_gate", "gate_temp"):
            assert token not in text, (index, token)


def test_every_number_on_a_slide_or_in_the_notes_came_from_data(deck):
    """Each numeral must have passed through Results.num (data) or Results.illus.

    Results.num only formats values read from tracked CSVs/records; a number
    typed into a slide string would not be in either set and fails here.
    """
    _, texts, notes, _, results = deck
    allowed = results.emitted | results.illustrative
    for index, text in enumerate(texts + notes, start=1):
        for token in number_tokens(text):
            if len(token) == 1:      # single digits: counts, indices, "exp(0) = 1"
                continue
            assert token in allowed, f"slide/notes {index}: '{token}' was not produced from data"


def test_illustrative_numbers_are_labelled_on_their_slide(deck):
    _, texts, _, _, results = deck
    only_illustrative = results.illustrative - results.emitted
    for index, text in enumerate(texts, start=1):
        tokens = {t for t in number_tokens(text) if len(t) > 1}
        if tokens & only_illustrative:
            assert "llustrative" in text, f"slide {index} shows illustrative numbers without saying so"


def test_placeholders_name_their_evidence(deck):
    _, texts, _, _, _ = deck
    found = set()
    for text in texts:
        for match in re.finditer(r"\[Pending ([^:\]]+):", text):
            assert match.group(1) in EVIDENCE_IDS, match.group(1)
            found.add(match.group(1))
    # every T05 item that has a home in the outline is on a slide
    assert {"A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "T06", "T07", "T08"} <= found


def test_fonts_are_projector_legible(deck):
    prs = deck[0]
    for index, slide in enumerate(prs.slides, start=1):
        for shape in slide.shapes:
            if shape.name == "footer":
                continue
            frames = []
            if shape.has_text_frame:
                frames.append(shape.text_frame)
            if shape.has_table:
                frames.extend(cell.text_frame for row in shape.table.rows for cell in row.cells)
            for frame in frames:
                for paragraph in frame.paragraphs:
                    for run in paragraph.runs:
                        if run.text.strip() and run.font.size is not None:
                            assert run.font.size >= Pt(14), (index, run.text[:40])


def test_headline_slide_quotes_the_summary_csv_means(deck):
    _, texts, _, _, _ = deck
    summary = pd.read_csv(ROOT / "results/summary.csv")
    means = summary.groupby("method")[["accuracy", "ece"]].mean()
    slide = next(t for t in texts if "Headline results" in t)
    for method in ("clipzs", "tda", "cal_tda_gate", "cal_tda_temp", "cal_tda_gate_temp"):
        assert f"{means.loc[method, 'accuracy']:.2f}" in slide, method
        assert f"{means.loc[method, 'ece']:.2f}" in slide, method


def test_gate_result_slide_matches_ablation_gate_csv(deck):
    _, texts, _, _, _ = deck
    gate = pd.read_csv(ROOT / "results/ablation_gate.csv")
    means = gate.groupby("variant")[["accuracy", "ece"]].mean()
    slide = next(t for t in texts if "Result: gated fusion" in t)
    assert f"{means.loc['gate', 'ece']:.2f}" in slide
    assert f"{means.loc['none', 'ece']:.2f}" in slide
    assert f"{means.loc['gate', 'accuracy']:.2f}" in slide
    summary = pd.read_csv(ROOT / "results/summary.csv").set_index(["dataset", "method"])
    beats = sum(gate.set_index(["dataset", "variant"]).loc[(d, "gate"), "ece"] < summary.loc[(d, "clipzs"), "ece"]
                for d in gate.dataset.unique())
    assert f"Beats CLIP's ECE on {beats} of 5" in slide


def test_margin_slide_shows_admission_rates_from_the_raw_records(deck):
    _, texts, _, _, _ = deck
    slide = next(t for t in texts if "Margin admission: a negative result" in t)
    for dataset in ("aircraft", "pets", "eurosat"):
        record = json.loads((ROOT / f"results/raw/ablation/{dataset}_margin.json").read_text(encoding="utf-8"))
        rate = 100.0 * record["hyperparams"]["admission"]["admission_rate"]
        assert f"{rate:.1f}%" in slide, dataset


def test_ablation_slide_lists_the_four_variants(deck):
    _, texts, _, _, _ = deck
    gate = pd.read_csv(ROOT / "results/ablation_gate.csv")
    means = gate.groupby("variant")[["accuracy", "ece"]].mean()
    slide = next(t for t in texts if "do not stack" in t)
    for variant in ("none", "gate", "temp", "gate_temp"):
        assert f"{means.loc[variant, 'ece']:.2f}" in slide, variant


def test_clip_vs_tda_slide_uses_reliability_stats(deck):
    _, texts, _, _, _ = deck
    rel = pd.read_csv(ROOT / "results/analysis/reliability_stats.csv")
    head = rel[rel.study == "headline"].set_index(["dataset", "variant"])
    slide = next(t for t in texts if "CLIP vs TDA: accuracy, confidence" in t)
    for dataset in ("dtd", "pets", "aircraft"):
        assert f"{head.loc[(dataset, 'tda'), 'mean_confidence']:.2f}" in slide
        assert f"{head.loc[(dataset, 'clipzs'), 'mean_confidence']:.2f}" in slide


def test_tda_hyperparameters_on_slides_match_the_run_records(deck):
    _, texts, _, _, _ = deck
    slide = next(t for t in texts if "What α and β mean" in t)
    for dataset in ("dtd", "eurosat", "aircraft"):
        hp = json.loads((ROOT / f"results/raw/{dataset}_tda.json").read_text(encoding="utf-8"))["hyperparams"]
        assert f"{hp['alpha']:.1f}" in slide and f"{hp['beta']:.1f}" in slide
