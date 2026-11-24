import xml.etree.ElementTree as ET

from compound_search.figure import AXIS_HI, AXIS_LO, GROUPS, PLOT_X0, PLOT_X1, render_svg, x_of
from compound_search.runs import DROPPED
from compound_search.splitter_speed import capitalised_words
from compound_search.tables import LABELS

NAMES = [name for _, names in GROUPS for name in names]


def summary(scores: dict[str, float] | None = None) -> dict:
    scores = scores or {}
    systems = {}
    for i, name in enumerate(NAMES):
        value = scores.get(name, 0.70 + 0.02 * i)
        systems[name] = {DROPPED: {"recall@10": value, "recall@10_ci95": [value - 0.03, value + 0.03]}}
    return {"n_queries": 1025, "systems": systems}


def test_axis_maps_its_ends_to_the_plot_edges():
    assert x_of(AXIS_LO) == PLOT_X0
    assert x_of(AXIS_HI) == PLOT_X1
    assert x_of((AXIS_LO + AXIS_HI) / 2) == (PLOT_X0 + PLOT_X1) / 2


def test_values_outside_the_axis_are_drawn_at_its_edge():
    assert x_of(0.2) == PLOT_X0
    assert x_of(1.3) == PLOT_X1


def test_chart_is_well_formed_and_has_one_dot_per_system():
    root = ET.fromstring(render_svg(summary()))
    ns = "{http://www.w3.org/2000/svg}"
    assert len(root.findall(f"{ns}circle")) == len(NAMES) == 10


def test_chart_shows_every_label_and_value_and_says_where_the_axis_starts():
    svg = render_svg(summary({"bm25_plain": 0.708}))
    texts = [t.text for t in ET.fromstring(svg).iter("{http://www.w3.org/2000/svg}text")]
    for name in NAMES:
        assert LABELS[name] in texts
    assert "70.8%" in texts
    assert "The axis starts at 60%." in texts
    assert "Recall@10 on the 1,025 held-out test questions, with 95% intervals" in texts


def test_a_better_system_is_drawn_further_right():
    svg = render_svg(summary({"bm25_plain": 0.70, "dense": 0.92}))
    cx = [float(c.get("cx")) for c in ET.fromstring(svg).iter("{http://www.w3.org/2000/svg}circle")]
    assert cx[NAMES.index("dense")] > cx[NAMES.index("bm25_plain")]


def test_speed_check_takes_each_capitalised_word_once_in_first_seen_order():
    assert capitalised_words(["Der Hund und der Hund", "Die Katze, der Hund"]) == ["Der", "Hund", "Die", "Katze"]
