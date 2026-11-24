"""Fetch the four Parquet files the study reads, at pinned revisions, and check them.

    python -m compound_search.download --raw data/raw          # download what is missing, then check
    python -I -m compound_search.download --raw data/raw --check   # only check the files that are there

The files are the auto-converted Parquet copies of `deepset/germandpr` and
`deepset/germanquad` on the Hugging Face Hub (both CC BY 4.0). The revisions are
pinned to the commits the study was run on, and every file is compared with its
SHA-256, so a later change upstream cannot silently change a result.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

HUB = "https://huggingface.co/datasets/deepset/{dataset}/resolve/{revision}/plain_text/{split}/0000.parquet"

# dataset -> (commit of the refs/convert/parquet branch, {split: sha256})
FILES: dict[str, tuple[str, dict[str, str]]] = {
    "germandpr": (
        "d0b867a49c45a44d172041351b5df73a87714372",
        {
            "train": "e8356a45e6bf41fee7007cf3f2444c7209351bf94baf71dac5f110a1117e09ff",
            "test": "f6106a22d6d6a6cab1621d70a06e0edd8edea9d1428e52424b1610f83b6228f0",
        },
    ),
    "germanquad": (
        "a2f3a59f0be843fc305d0417d7292ef0b1a66884",
        {
            "train": "319334092605ed9ade8a1e6535584debb2e99def474697417a06ad4a455bf0ec",
            "test": "c154513e016e73e4e32b46b48010f24a435ad7b0e96cc3858cfc20c7d29649bc",
        },
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def targets(raw: Path) -> list[tuple[Path, str, str]]:
    """(local path, url, expected sha256) for every file of the study."""
    out = []
    for dataset, (revision, splits) in FILES.items():
        for split, digest in splits.items():
            url = HUB.format(dataset=dataset, revision=revision, split=split)
            out.append((Path(raw) / dataset / f"{split}.parquet", url, digest))
    return out


def check(raw: Path) -> list[str]:
    """Problems found in the local files: one line per missing or changed file."""
    problems = []
    for path, _, digest in targets(raw):
        if not path.exists():
            problems.append(f"missing: {path}")
        elif sha256(path) != digest:
            problems.append(f"checksum differs: {path}")
    return problems


def download(raw: Path) -> None:
    for path, url, _ in targets(raw):
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}")
        partial = path.with_suffix(".part")
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "compound-search"})) as resp:
            partial.write_bytes(resp.read())
        partial.rename(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="do not download, only check the files")
    args = parser.parse_args()
    if not args.check:
        download(args.raw)
    problems = check(args.raw)
    for line in problems:
        print(line, file=sys.stderr)
    if problems:
        raise SystemExit(1)
    print(f"{len(targets(args.raw))} files present, checksums match")


if __name__ == "__main__":
    main()
