"""The data check: missing and changed files are reported, nothing is fetched in tests."""

from __future__ import annotations

from compound_search import download


def test_targets_cover_both_datasets_and_splits(tmp_path):
    paths = [path.relative_to(tmp_path).as_posix() for path, _, _ in download.targets(tmp_path)]
    assert sorted(paths) == [
        "germandpr/test.parquet",
        "germandpr/train.parquet",
        "germanquad/test.parquet",
        "germanquad/train.parquet",
    ]
    for _, url, digest in download.targets(tmp_path):
        assert "/resolve/" in url and "refs" not in url  # a pinned commit, not a moving branch
        assert len(digest) == 64


def test_check_reports_missing_and_changed_files(tmp_path, monkeypatch):
    assert len(download.check(tmp_path)) == 4
    good = tmp_path / "germandpr" / "test.parquet"
    good.parent.mkdir(parents=True)
    good.write_bytes(b"abc")
    assert f"checksum differs: {good}" in download.check(tmp_path)

    revision, splits = download.FILES["germandpr"]
    monkeypatch.setitem(download.FILES, "germandpr", (revision, {**splits, "test": download.sha256(good)}))
    problems = download.check(tmp_path)
    assert len(problems) == 3
    assert not any(str(good) in line for line in problems)
