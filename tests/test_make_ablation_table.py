from analysis.make_ablation_table import STUDIES


def test_gate_study_caption_does_not_number_the_ablations():
    # The report no longer presents two ablations; the margin-off study is the
    # ablation, so its caption must not call itself "second".
    caption = STUDIES["gate"]["caption"].lower()
    assert "second" not in caption
    assert "first" not in caption
    assert "margin admission off" in caption
