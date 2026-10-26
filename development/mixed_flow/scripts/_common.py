"""Shared plotting/configuration helpers for the standalone examples."""

from dataclasses import MISSING, asdict, fields
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from development.mixed_flow.meridional import (
    BladeEdges,
    BladeMeridional,
    MeanlineDimensions,
    MeridionalChannel,
    WallShape,
)

STUDY = Path(__file__).resolve().parents[1]
HUB, SHROUD, BLADE, SPLITTER = "#176b89", "#d16b27", "#778ca3", "#8752a1"


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#253747",
            "text.color": "#253747",
            "axes.edgecolor": "#80909c",
            "grid.alpha": 0.18,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
        }
    )


def example_config():
    return {
        "dimensions": asdict(MeanlineDimensions.from_ratios()),
        "hub": asdict(WallShape()),
        "shroud": asdict(WallShape()),
        "blade": asdict(BladeEdges()),
        "splitter_start": 0.25,
    }


def build(config=None):
    data = example_config() if config is None else config
    channel = MeridionalChannel(
        MeanlineDimensions(**data["dimensions"]),
        hub=WallShape(**data.get("hub", {})),
        shroud=WallShape(**data.get("shroud", {})),
    )
    blade = BladeMeridional(channel, BladeEdges(**data.get("blade", {})))
    splitter = blade.splitter(data.get("splitter_start", 0.25))
    return channel, blade, splitter


def load_config(path):
    """Read a YAML engineering-input mapping with explicit structural errors."""
    if path is None:
        return example_config()
    path = Path(path)
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("Configuration files must use the .yaml or .yml extension.")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(
            f"Could not read YAML configuration {path.name}: {error}"
        ) from error
    if not isinstance(data, dict):
        raise ValueError(
            "The YAML configuration must contain a mapping of parameter groups."
        )
    unknown = set(data) - {"dimensions", "hub", "shroud", "blade", "splitter_start"}
    if unknown:
        raise ValueError(
            f"Unknown configuration keys: {', '.join(sorted(map(str, unknown)))}."
        )
    if "dimensions" not in data:
        raise ValueError("The YAML configuration requires a dimensions mapping.")
    for name, model in (
        ("dimensions", MeanlineDimensions),
        ("hub", WallShape),
        ("shroud", WallShape),
        ("blade", BladeEdges),
    ):
        group = data.get(name, {})
        if not isinstance(group, dict):
            raise ValueError(f"{name} must be a mapping; use {{}} for its defaults.")
        specifications = fields(model)
        unknown = set(group) - {field.name for field in specifications}
        if unknown:
            raise ValueError(
                f"Unknown {name} parameters: {', '.join(sorted(map(str, unknown)))}."
            )
        missing = {
            field.name
            for field in specifications
            if field.default is MISSING and field.default_factory is MISSING
        } - set(group)
        if missing:
            raise ValueError(
                f"Missing {name} parameters: {', '.join(sorted(missing))}."
            )
    return data


def line(ax, points, **kwargs):
    """Plot SI geometry in millimetres."""
    return ax.plot(points[..., 0] * 1000, points[..., 1] * 1000, **kwargs)


def draw_channel(ax, channel, blade=None, splitter=None, stations=True):
    s = np.linspace(0, 1, 301)
    eta = np.linspace(0, 1, 101)
    if blade is not None:
        boundary = np.concatenate(
            [
                blade.point(s, 0),
                blade.trailing_edge(eta),
                blade.point(s[::-1], 1),
                blade.leading_edge(eta[::-1]),
            ]
        )
        ax.fill(
            boundary[:, 0] * 1000,
            boundary[:, 1] * 1000,
            color=BLADE,
            alpha=0.15,
            label="Blade meridional extent",
        )
    if stations:
        for position in np.linspace(0, 1, 9):
            line(ax, channel.point(position, eta), color="#b5c0c8", lw=0.7)
    line(ax, channel.hub.point_at(s), color=HUB, lw=2.5, label="Hub")
    line(ax, channel.shroud.point_at(s), color=SHROUD, lw=2.5, label="Shroud")
    if blade is not None:
        line(ax, blade.leading_edge(eta), color="#354859", lw=1.8, label="Main LE / TE")
        line(ax, blade.trailing_edge(eta), color="#354859", lw=1.8)
    if splitter is not None:
        line(
            ax,
            splitter.leading_edge(eta),
            color=SPLITTER,
            lw=1.8,
            ls="--",
            label="Splitter LE",
        )
    ax.set(xlabel="Axial coordinate z [mm]", ylabel="Radius r [mm]", aspect="equal")
    ax.grid(True)


def save(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "png"):
        fig.savefig(directory / f"{name}.{suffix}", dpi=170, bbox_inches="tight")
    print(directory / f"{name}.svg")
