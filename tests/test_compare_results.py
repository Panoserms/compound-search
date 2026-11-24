import json

from compound_search.compare_results import compare_folders, differences


def test_equal_reports_have_no_differences():
    a = {"system": "x", "pools": {"p": {"gold_rank": {"q1": 3, "q2": None}, "summary": {"recall@10": 0.5}}}}
    assert differences(a, json.loads(json.dumps(a))) == []


def test_a_changed_rank_is_named_with_its_path():
    a = {"pools": {"p": {"gold_rank": {"q1": 3}}}}
    b = {"pools": {"p": {"gold_rank": {"q1": 4}}}}
    assert differences(a, b) == ["pools.p.gold_rank.q1: 3 != 4"]


def test_timing_fields_are_ignored():
    a = {"seconds": 1.5, "timing_seconds": {"index_build": 2.0}, "value": 1}
    b = {"seconds": 9.9, "timing_seconds": {"index_build": 7.0}, "value": 1}
    assert differences(a, b) == []


def test_missing_and_extra_keys_are_reported():
    assert differences({"a": 1, "b": 2}, {"a": 1, "c": 3}) == ["b: only in the first", "c: only in the second"]


def test_lists_are_compared_item_by_item():
    assert differences({"ci": [0.1, 0.2]}, {"ci": [0.1, 0.3]}) == ["ci[1]: 0.2 != 0.3"]
    assert differences({"ci": [0.1]}, {"ci": [0.1, 0.3]}) == ["ci: length 1 != 2"]


def test_folders_are_compared_file_by_file(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    (first / "dev").mkdir(parents=True)
    (second / "dev").mkdir(parents=True)
    (first / "dev" / "x.json").write_text('{"v": 1}')
    (second / "dev" / "x.json").write_text('{"v": 2}')
    (first / "same.json").write_text('{"v": 1, "seconds": 3}')
    (second / "same.json").write_text('{"v": 1, "seconds": 4}')
    (first / "only_first.json").write_text("{}")
    (first / "notes.md").write_text("not compared")
    result = compare_folders(first, second)
    assert result["identical"] == ["same.json"]
    assert result["different"] == {"dev/x.json": ["v: 1 != 2"]}
    assert result["only_in_first"] == ["only_first.json"]
    assert result["only_in_second"] == []
