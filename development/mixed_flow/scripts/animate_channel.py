"""Animate independent engineering controls using the real meridional kernel.

The default produces a portable looping GIF and a poster. MP4 is optional and
requires an FFmpeg build with libx264 (``--format mp4`` or ``--format both``).
Every chapter smoothly moves from the baseline to one changed value and back.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np
from PIL import Image
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
INK = "#263d4a"
MUTED = "#68808f"
HUB = "#176b89"
SHROUD = "#d16b27"
SPLITTER = "#8752a1"
BASELINE = "#bdc9d1"


@dataclass(frozen=True)
class Chapter:
    title: str
    group: str
    field: str
    baseline: float
    target: float
    label: str
    units: str
    note: str
    scale: float = 1.0


CHAPTERS = (
    Chapter(
        "Channel proportions",
        "dimensions",
        "axial_length",
        0.060,
        0.080,
        "Axial length",
        "mm",
        "Change the axial distance between the inlet and outlet stations.",
        1000,
    ),
    Chapter(
        "Outlet passage width",
        "dimensions",
        "outlet_width",
        0.008,
        0.014,
        "Outlet width",
        "mm",
        "The outlet radius stays fixed while the discharge passage opens.",
        1000,
    ),
    Chapter(
        "Inclined inlet",
        "dimensions",
        "inlet_flow_angle_deg",
        0,
        25,
        "Inlet inclination",
        "°",
        "The inlet station and inherited wall tangents rotate together.",
    ),
    Chapter(
        "Mixed-flow outlet",
        "dimensions",
        "outlet_flow_angle_deg",
        90,
        55,
        "Outlet inclination",
        "°",
        "Move continuously from a radial outlet towards mixed flow.",
    ),
    Chapter(
        "Independent hub slope",
        "hub",
        "inlet_angle_deg",
        0,
        20,
        "Hub inlet tangent",
        "°",
        "Only the hub wall changes; its inlet and outlet points stay fixed.",
    ),
    Chapter(
        "Shroud turning distribution",
        "shroud",
        "outlet_tangent_fraction",
        0.45,
        0.75,
        "Shroud outlet tangent reach",
        "",
        "Extend the influence of the outlet tangent without moving the endpoints.",
    ),
    Chapter(
        "Independent shroud bow",
        "shroud",
        "bow",
        0,
        0.050,
        "Shroud bow / chord",
        "",
        "Add normal displacement while preserving endpoint positions and slopes.",
    ),
    Chapter(
        "Blade leading-edge sweep",
        "blade",
        "leading_hub",
        0,
        0.27,
        "Hub leading-edge fraction",
        "",
        "Move the hub leading edge downstream; the channel walls do not change.",
    ),
    Chapter(
        "Bowed leading edge",
        "blade",
        "leading_bow",
        0,
        0.20,
        "Leading-edge bow",
        "",
        "Move the edge interior downstream while keeping both wall intersections fixed.",
    ),
    Chapter(
        "Independent blade cutback",
        "blade",
        "trailing_hub",
        1,
        0.80,
        "Hub trailing-edge fraction",
        "",
        "Trim the blade inside the existing channel without changing the casing.",
    ),
    Chapter(
        "Splitter placement",
        "splitter",
        "start_fraction",
        0.25,
        0.50,
        "Splitter start fraction",
        "",
        "Shorten the splitter from its inlet; its trailing edge stays with the main blade.",
    ),
)


def excursion(phase: float) -> float:
    """A quiet start/end and a smooth baseline-to-target-to-baseline motion."""
    if phase <= 0.08 or phase >= 0.92:
        return 0.0
    if phase < 0.43:
        x = (phase - 0.08) / 0.35
        return 0.5 - 0.5 * np.cos(np.pi * x)
    if phase <= 0.57:
        return 1.0
    x = (phase - 0.57) / 0.35
    return 0.5 + 0.5 * np.cos(np.pi * x)


def build_frame(chapter: Chapter, phase: float):
    """Build one real, validated geometry; no data from the app is required."""
    value = chapter.baseline + excursion(phase) * (chapter.target - chapter.baseline)
    dims = MeanlineDimensions.from_ratios()
    hub, shroud, edges = WallShape(), WallShape(), BladeEdges()
    fraction = 0.25
    if chapter.group == "dimensions":
        dims = replace(dims, **{chapter.field: value})
    elif chapter.group == "hub":
        hub = replace(hub, **{chapter.field: value})
    elif chapter.group == "shroud":
        shroud = replace(shroud, **{chapter.field: value})
    elif chapter.group == "blade":
        edges = replace(edges, **{chapter.field: value})
    else:
        fraction = value
    channel = MeridionalChannel(dims, hub=hub, shroud=shroud)
    blade = BladeMeridional(channel, edges)
    return value, channel, blade, blade.splitter(fraction)


def find_ffmpeg(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    candidate = shutil.which("ffmpeg")
    if candidate:
        return candidate
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def create_canvas(width: int, height: int):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "axes.edgecolor": "#ced9e0",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#f3f6f8",
            "axes.facecolor": "white",
            "grid.color": "#e7edf1",
            "grid.linewidth": 0.7,
        }
    )
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    fig.text(0.045, 0.945, "MERIDIONAL GEOMETRY LAB", color=HUB, size=11, weight="bold")
    title = fig.text(0.045, 0.888, CHAPTERS[0].title, size=24, weight="bold")
    subtitle = fig.text(0.045, 0.842, CHAPTERS[0].note, color=MUTED, size=10)
    number = fig.text(0.955, 0.945, "01 / 11", ha="right", color=MUTED, size=11)
    ax = fig.add_axes([0.065, 0.17, 0.45, 0.60])
    ax.set(
        xlim=(-11, 94),
        ylim=(17, 108),
        xlabel="Axial coordinate z [mm]",
        ylabel="Radius r [mm]",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax_area = fig.add_axes([0.61, 0.19, 0.335, 0.26])
    ax_area.set(
        xlim=(0, 1),
        ylim=(3500, 13000),
        xlabel="Normalized wall arclength s",
        ylabel="Geometric area [mm²]",
    )
    ax_area.grid(True)
    ax_area.set_title(
        "Channel area response", loc="left", pad=13, size=12, weight="bold"
    )
    fig.text(
        0.61, 0.675, "ACTIVE ENGINEERING CONTROL", color=MUTED, size=9, weight="bold"
    )
    control = fig.text(0.61, 0.635, "Axial length", size=14)
    value_text = fig.text(0.61, 0.565, "60.0 mm", size=30, weight="bold", color=HUB)
    group_text = fig.text(0.61, 0.515, "Channel dimensions", color=MUTED, size=10)
    fig.text(
        0.045,
        0.072,
        "Gray dashed: baseline   ·   One control changes at a time   ·   Fixed drawing scale",
        color=MUTED,
        size=9,
    )
    fig.text(
        0.045,
        0.038,
        "Illustrative geometry; area excludes blade blockage and is not the blade-passage throat.",
        color=MUTED,
        size=8,
    )
    progress_ax = fig.add_axes([0.045, 0.012, 0.91, 0.007])
    progress_ax.set(xlim=(0, 1), ylim=(0, 1))
    progress_ax.set_axis_off()
    progress_ax.add_patch(Rectangle((0, 0), 1, 1, color="#dce6ed"))
    progress = progress_ax.add_patch(Rectangle((0, 0), 0, 1, color=HUB))

    _, baseline, main, split = build_frame(CHAPTERS[0], 0)
    s, eta = np.linspace(0, 1, 251), np.linspace(0, 1, 81)
    d0 = baseline.diagnostics(201)
    for wall in (baseline.hub, baseline.shroud):
        points = wall.point_at(s) * 1000
        ax.plot(*points.T, color=BASELINE, ls="--", lw=1.5, zorder=2)
    for edge in (
        main.leading_edge(eta),
        main.trailing_edge(eta),
        split.leading_edge(eta),
    ):
        ax.plot(*(edge * 1000).T, color=BASELINE, ls="--", lw=1, zorder=2)
    ax_area.plot(d0["s"], d0["area"] * 1e6, color=BASELINE, ls="--", lw=1.5)
    fill = ax.add_patch(
        Polygon(
            np.zeros((4, 2)),
            closed=True,
            facecolor="#dee8ee",
            alpha=0.65,
            edgecolor="none",
            zorder=1,
        )
    )
    stations = [ax.plot([], [], color="#cbd6de", lw=0.7, zorder=1)[0] for _ in range(7)]
    (hub_line,) = ax.plot([], [], color=HUB, lw=2.8, label="Hub", zorder=4)
    (shroud_line,) = ax.plot([], [], color=SHROUD, lw=2.8, label="Shroud", zorder=4)
    (leading,) = ax.plot([], [], color=INK, lw=1.8, label="Main blade edges", zorder=5)
    (trailing,) = ax.plot([], [], color=INK, lw=1.8, zorder=5)
    (splitter,) = ax.plot(
        [], [], color=SPLITTER, lw=2, ls="--", label="Splitter leading edge", zorder=5
    )
    (area,) = ax_area.plot([], [], color=HUB, lw=2.1)
    ax.legend(
        loc="upper left",
        fontsize=9,
        frameon=True,
        facecolor="white",
        edgecolor="#e1e8ed",
    )
    return fig, {
        "title": title,
        "subtitle": subtitle,
        "number": number,
        "control": control,
        "value": value_text,
        "group": group_text,
        "progress": progress,
        "fill": fill,
        "stations": stations,
        "hub": hub_line,
        "shroud": shroud_line,
        "leading": leading,
        "trailing": trailing,
        "splitter": splitter,
        "area": area,
    }


def render_frame(fig, artists, chapter_index, phase, progress):
    chapter = CHAPTERS[chapter_index]
    value, channel, blade, splitter = build_frame(chapter, phase)
    s, eta = np.linspace(0, 1, 251), np.linspace(0, 1, 81)
    d = channel.diagnostics(201)
    points = {
        "hub": channel.hub.point_at(s),
        "shroud": channel.shroud.point_at(s),
        "leading": blade.leading_edge(eta),
        "trailing": blade.trailing_edge(eta),
        "splitter": splitter.leading_edge(eta),
    }
    for key, data in points.items():
        artists[key].set_data(*(data * 1000).T)
    boundary = np.concatenate(
        [
            blade.point(s, 0),
            points["trailing"],
            blade.point(s[::-1], 1),
            points["leading"][::-1],
        ]
    )
    artists["fill"].set_xy(boundary * 1000)
    for line, position in zip(artists["stations"], np.linspace(0, 1, 7)):
        line.set_data(*(channel.point(position, eta) * 1000).T)
    artists["area"].set_data(d["s"], d["area"] * 1e6)
    artists["title"].set_text(chapter.title)
    artists["subtitle"].set_text(chapter.note)
    artists["number"].set_text(f"{chapter_index + 1:02d} / {len(CHAPTERS):02d}")
    artists["control"].set_text(chapter.label)
    value_format = ".1f" if chapter.units else ".3f"
    artists["value"].set_text(
        f"{value * chapter.scale:{value_format}}{(' ' + chapter.units) if chapter.units else ''}"
    )
    group_label = {
        "dimensions": "Channel dimensions",
        "hub": "Hub wall only",
        "shroud": "Shroud wall only",
        "blade": "Blade extent · walls held fixed",
        "splitter": "Splitter extent · walls and main blade held fixed",
    }
    artists["group"].set_text(group_label[chapter.group])
    artists["progress"].set_width(progress)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(rgba[:, :, :3].copy(), mode="RGB")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=STUDY / "media")
    parser.add_argument(
        "--seconds",
        type=float,
        default=44,
        help="Total duration; equal time for each control",
    )
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--gif-width", type=int, default=960)
    parser.add_argument("--format", choices=("gif", "mp4", "both"), default="gif")
    parser.add_argument(
        "--ffmpeg", help="FFmpeg binary with libx264 for optional MP4 output"
    )
    args = parser.parse_args(argv)
    if args.seconds < len(CHAPTERS) or not 1 <= args.fps <= 60:
        parser.error("Use at least one second per chapter and 1–60 frames/second.")
    if round(args.seconds * args.fps / len(CHAPTERS)) < 8:
        parser.error(
            "Use at least eight frames per chapter so the parameter motion remains visible."
        )
    if min(args.width, args.height, args.gif_width) < 320:
        parser.error("Image dimensions must be at least 320 pixels.")
    if args.width % 2 or args.height % 2:
        parser.error("Width and height must be even for MP4 compatibility.")
    ffmpeg = find_ffmpeg(args.ffmpeg) if args.format in ("mp4", "both") else None
    if args.format in ("mp4", "both") and ffmpeg is None:
        parser.error(
            "MP4 needs FFmpeg with libx264. Install FFmpeg or imageio-ffmpeg, or use --format gif."
        )
    if ffmpeg:
        encoders = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        if "libx264" not in encoders:
            parser.error(
                "The selected FFmpeg lacks libx264; use a full build or --format gif."
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    count = max(len(CHAPTERS), round(args.seconds * args.fps))
    frames_per_chapter = max(2, round(count / len(CHAPTERS)))
    count = len(CHAPTERS) * frames_per_chapter
    fig, artists = create_canvas(args.width, args.height)
    gif_frames = []
    process = None
    video_path = args.output_dir / "meridional_parameter_tour.mp4"
    if ffmpeg:
        process = subprocess.Popen(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{args.width}x{args.height}",
                "-r",
                str(args.fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(video_path),
            ],
            stdin=subprocess.PIPE,
        )
    # A stable palette prevents antialiased labels from changing colour frame to frame.
    palette = None
    try:
        for index in range(count):
            chapter_index, chapter_frame = divmod(index, frames_per_chapter)
            phase = chapter_frame / (frames_per_chapter - 1)
            frame = render_frame(
                fig, artists, chapter_index, phase, index / (count - 1)
            )
            if index == 0:
                frame.save(args.output_dir / "meridional_parameter_tour_poster.png")
            if chapter_frame == frames_per_chapter // 2 and chapter_index in (3, 7):
                frame.save(
                    args.output_dir
                    / f"meridional_tour_chapter_{chapter_index + 1:02d}.png"
                )
            if process is not None:
                process.stdin.write(frame.tobytes())
            if args.format in ("gif", "both"):
                if frame.width != args.gif_width:
                    frame = frame.resize(
                        (
                            args.gif_width,
                            round(args.gif_width * args.height / args.width),
                        ),
                        Image.Resampling.LANCZOS,
                    )
                if palette is None:
                    palette = frame.quantize(
                        colors=128, method=Image.Quantize.MEDIANCUT
                    )
                gif_frames.append(
                    frame.quantize(palette=palette, dither=Image.Dither.NONE)
                )
            if chapter_frame == frames_per_chapter - 1:
                print(
                    f"Rendered {chapter_index + 1:02d}/{len(CHAPTERS):02d}: {CHAPTERS[chapter_index].title}",
                    flush=True,
                )
    finally:
        plt.close(fig)
        if process is not None:
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("FFmpeg could not encode the animation.")

    gif_path = args.output_dir / "meridional_parameter_tour.gif"
    if gif_frames:
        # GIF stores centiseconds; distribute quantization so total duration is retained.
        boundaries = np.rint(np.arange(count + 1) * 100 / args.fps).astype(int)
        durations = np.diff(boundaries) * 10
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=durations.tolist(),
            loop=0,
            optimize=True,
            disposal=1,
        )
        with Image.open(gif_path) as saved:
            actual_frames = saved.n_frames
            duration_ms = 0
            for i in range(actual_frames):
                saved.seek(i)
                duration_ms += saved.info["duration"]
                saved.load()
        if abs(duration_ms / 1000 - count / args.fps) > 0.02:
            raise RuntimeError(
                "Encoded GIF duration differs from the requested frame timing."
            )
        print(
            f"GIF verified: {actual_frames} encoded frames, {duration_ms / 1000:.2f} s, {gif_path.stat().st_size / 1e6:.2f} MB"
        )

    baseline = {
        "dimensions": asdict(MeanlineDimensions.from_ratios()),
        "hub": asdict(WallShape()),
        "shroud": asdict(WallShape()),
        "blade": asdict(BladeEdges()),
        "splitter_start": 0.25,
    }
    manifest = {
        "title": "Meridional channel engineering parameter tour",
        "duration_seconds": count / args.fps,
        "frames_per_second": args.fps,
        "rendered_frames": count,
        "video_resolution": [args.width, args.height],
        "gif_resolution": [
            args.gif_width,
            round(args.gif_width * args.height / args.width),
        ],
        "formats": [name for name in ("gif", "mp4") if args.format in (name, "both")],
        "length_units": "m",
        "angle_units": "degrees from positive axial direction",
        "baseline": baseline,
        "motion": "Each chapter holds baseline, smoothly reaches the target, holds, and returns to baseline.",
        "chapters": [
            {
                "start_seconds": i * frames_per_chapter / args.fps,
                "end_seconds": (i + 1) * frames_per_chapter / args.fps,
                **asdict(chapter),
            }
            for i, chapter in enumerate(CHAPTERS)
        ],
        "notes": [
            "Every frame is evaluated and validated by the meridional geometry kernel.",
            "Blade edge and splitter fractions use the channel s coordinate, which is normalized wall arclength at hub and shroud.",
            "Geometric area excludes blade blockage and is not the blade-passage throat.",
        ],
    }
    manifest_path = args.output_dir / "meridional_parameter_tour.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    if args.format in ("gif", "both"):
        print(gif_path)
    if args.format in ("mp4", "both"):
        print(video_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
