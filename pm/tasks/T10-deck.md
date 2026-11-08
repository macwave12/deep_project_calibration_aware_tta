# T10 — Rebuild the deck

Status: draft-review · Owner: deck agent · Depends on: T05–T09

Rewrite `presentation/build_pptx.py` to the outline in `PLAN.md`, pulling every number from
CSVs, with speaker notes in `slides.md`. Address every point in Idan's review of
2026-11-07 (listed in `pm/research/deck-review-2026-11-07.md`). Render to PDF/PNG for
review. Keep `tests/test_build_pptx.py` green and extend it to the new slides.

Progress: draft v2 (2026-11-08) built from the existing CSVs with labelled placeholders for
T05 A2–A10, T06, T07, T08 — see `pm/handoffs/2026-11-08-T10-deck-v2-draft.md`. Awaiting
Idan's structure review; the evidence pass fills the placeholders once features arrive.
