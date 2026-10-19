"""Build the offline review with embedded, editable decision metadata."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    decisions = json.loads((ROOT / "decisions.json").read_text(encoding="utf-8"))
    serialized = json.dumps(decisions, ensure_ascii=False).replace("<", "\\u003c")
    template = (ROOT / "review.template.html").read_text(encoding="utf-8")
    (ROOT / "index.html").write_text(template.replace("__DECISIONS__", serialized), encoding="utf-8")
    print(ROOT / "index.html")


if __name__ == "__main__":
    main()
