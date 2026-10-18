"""Compatibility entry point for the complete stage sensitivity studies.

The common study.py writes samples into each model folder and report.py
writes separate complete reports. Includes KO, Benner and isentropic models; supports --models, --render-only and --check.
"""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from study import main

if __name__ == "__main__":
    main()
