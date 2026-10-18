"""Compatibility entry point for the consolidated, lighter shock study.

The retained analyses now run together in run.py. The former separate floor,
pressure-head and full hub/tip shock sweeps have been removed. The options
--render-only and --verify-only are forwarded unchanged.
"""
from run import main


if __name__ == "__main__":
    main()
