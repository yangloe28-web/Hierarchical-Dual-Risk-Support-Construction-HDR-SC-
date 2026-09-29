from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_TOP_LEVEL = {
    "artifacts",
    "automation",
    "figures",
    "outputs",
    "reports",
    "results",
    "tmp",
}
FORBIDDEN_SUFFIXES = {
    ".doc",
    ".docx",
    ".pdf",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".pt",
    ".pth",
    ".npz",
}
IGNORED_PARTS = {".git", ".pytest_cache", ".venv", "__pycache__"}
WINDOWS_ABSOLUTE = re.compile(r"(?i)(?:^|[\s'\"])[a-z]:[\\/]")
SECRET_PATTERNS = {
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "GitHub token": re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    "Private key": re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    ),
}


def main() -> int:
    problems: list[str] = []
    present_top_level = {path.name.lower() for path in ROOT.iterdir()}
    for name in sorted(FORBIDDEN_TOP_LEVEL.intersection(present_top_level)):
        problems.append(f"forbidden top-level directory: {name}")

    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in IGNORED_PARTS for part in path.parts):
            continue
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"forbidden binary or manuscript asset: {relative}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            problems.append(f"unreviewed binary file: {relative}")
            continue
        if path != Path(__file__).resolve() and WINDOWS_ABSOLUTE.search(text):
            problems.append(f"absolute Windows path: {relative}")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                problems.append(f"{label}: {relative}")

    if problems:
        print("Release audit failed:")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print(
        "Release audit passed: no private paths, common secrets, "
        "or paper assets found."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

