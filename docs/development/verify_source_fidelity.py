"""Verify preserved note hashes and the complete Sphinx rendering of those notes.

The canonical notes live in docs/source/notes, with the unpublished placeholder
in docs/development. Their recorded hashes preserve the original baseline after
the duplicate copies formerly inside the Python package have been removed.
"""

from pathlib import Path
import argparse
import base64
import hashlib
import json
import re


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("source_fidelity.json")


def rendering_wrapper(entry, original):
    """Partition complete notes at math boundaries without editing their text.

    MyST does not interrupt an existing paragraph at ``$$``. Starting a new
    include there separates original prose from the display block. Original
    TeX bodies use ``:nowrap:`` to prevent Sphinx from injecting align/split
    environments around existing gather environments and blank lines. SVG data
    URLs are rejected by Markdown's link validator, so their exact decoded bytes
    become image assets.
    """
    lines = original.decode("utf-8").replace("\r\n", "\n").splitlines(keepends=True)
    starts = {0}
    math_start = None
    maths = {}
    images = {}
    for index, line in enumerate(lines):
        if line.strip() == "$$":
            if math_start is None:
                math_start = index
                starts.add(index)
            else:
                maths[math_start] = "".join(lines[math_start + 1:index])
                starts.add(index + 1)
                math_start = None
        image = re.fullmatch(
            r"!\[([^\]]*)\]\(data:image/svg\+xml;base64,([^)]+)\)\s*",
            line,
        )
        if image:
            starts.update((index, index + 1))
            images[index] = (image[1], base64.b64decode(image[2]))
    if math_start is not None:
        raise AssertionError(f"Unpaired display delimiters: {entry['source']}")
    starts.add(len(lines))
    boundaries = sorted(starts)
    output = entry.get("page_title", "")
    assets = {}
    for first, last in zip(boundaries, boundaries[1:]):
        if first in maths:
            output += "```{math}\n:nowrap:\n\n\\[\n" + maths[first] + "\\]\n```\n\n"
        elif first in images:
            alt, svg = images[first]
            name = f"{Path(entry['source']).stem}_{first + 1}.svg"
            assets[name] = svg
            output += f"```{{image}} ../assets/{name}\n:alt: {alt}\n```\n\n"
        else:
            output += (
                f"```{{include}} ../notes/{Path(entry['source']).name}\n"
                f":start-line: {first}\n:end-line: {last}\n```\n\n"
            )
    return output, assets


def verify(write_wrappers=False):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures = []
    for entry in manifest["files"]:
        original = (ROOT / entry["source"]).read_bytes()
        normalized = original.replace(b"\r\n", b"\n")
        if hashlib.sha256(normalized).hexdigest() != entry["sha256_lf"]:
            failures.append(f"Original source hash changed: {entry['source']}")
        if entry.get("page"):
            expected, assets = rendering_wrapper(entry, original)
            page = ROOT / entry["page"]
            if write_wrappers:
                page.write_text(expected, encoding="utf-8")
            wrapper = page.read_text(encoding="utf-8")
            if wrapper != expected:
                failures.append(f"Verbatim include wrapper changed: {entry['page']}")
            for name, svg in assets.items():
                asset = ROOT / "docs" / "source" / "assets" / name
                if write_wrappers:
                    asset.parent.mkdir(parents=True, exist_ok=True)
                    asset.write_bytes(svg)
                if asset.read_bytes() != svg:
                    failures.append(f"Original SVG image changed: {name}")
    if failures:
        raise AssertionError("\n".join(failures))
    print(
        f"Verified {len(manifest['files'])} preserved original-note hashes, "
        "complete rendering wrappers, and original SVG bytes."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-wrappers", action="store_true")
    verify(parser.parse_args().write_wrappers)
