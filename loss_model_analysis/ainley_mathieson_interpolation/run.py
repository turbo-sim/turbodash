"""Reproduce the implemented profile blend, or --render-only / --verify-only."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--render-only", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.render_only:
        from study import calculate, verify
        if not args.verify_only:
            calculate()
        result = verify()
        print(json.dumps({k: v for k, v in result.items() if k != "zero_crossings"}, indent=2), flush=True)
    if not args.verify_only:
        from plot import render
        print(json.dumps(render()), flush=True)


if __name__ == "__main__":
    main()
