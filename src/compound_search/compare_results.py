"""Compare two results folders, JSON file by JSON file. Used to check a reproduction.

    python -I -m compound_search.compare_results results /path/to/other/results

Timing fields are ignored, everything else must be equal. Exit code 1 if anything differs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

IGNORED_KEYS = frozenset({"seconds", "timing_seconds"})


def differences(a, b, path: str = "") -> list[str]:
    """Every place where two JSON values differ, as 'path: a != b'."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            if key in IGNORED_KEYS:
                continue
            where = f"{path}.{key}" if path else str(key)
            if key not in b:
                out.append(f"{where}: only in the first")
            elif key not in a:
                out.append(f"{where}: only in the second")
            else:
                out.extend(differences(a[key], b[key], where))
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(differences(x, y, f"{path}[{i}]"))
        return out
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def compare_folders(first: Path, second: Path) -> dict:
    names_first = {p.relative_to(first).as_posix() for p in first.rglob("*.json")}
    names_second = {p.relative_to(second).as_posix() for p in second.rglob("*.json")}
    result: dict = {
        "identical": [],
        "different": {},
        "only_in_first": sorted(names_first - names_second),
        "only_in_second": sorted(names_second - names_first),
    }
    for name in sorted(names_first & names_second):
        diffs = differences(
            json.loads((first / name).read_text(encoding="utf-8")),
            json.loads((second / name).read_text(encoding="utf-8")),
        )
        if diffs:
            result["different"][name] = diffs
        else:
            result["identical"].append(name)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--out", type=Path, help="write the comparison as JSON")
    args = parser.parse_args()
    result = compare_folders(args.first, args.second)
    print(f"identical: {len(result['identical'])} files")
    for name, diffs in result["different"].items():
        print(f"DIFFERENT {name}: {len(diffs)} places")
        for line in diffs[:8]:
            print(f"    {line}")
    for key in ("only_in_first", "only_in_second"):
        if result[key]:
            print(f"{key}: {result[key]}")
    if args.out:
        args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if result["different"] or result["only_in_first"] or result["only_in_second"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
