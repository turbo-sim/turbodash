"""Plot the implemented blend and full thickness response from saved CSVs."""
import csv
import hashlib
import json
import os
from pathlib import Path
BASE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00")
SPECS = (
    ("vary_alpha2", "alpha2_deg", "z", "blended_reference", "blend_alpha2",
     r"Reference blend: varying $\alpha_2$, fixed $s/c=0.7$"),
    ("vary_pitch_chord", "pitch_chord", "z", "blended_reference", "blend_pitch_chord",
     r"Reference blend: varying $s/c$, fixed $\alpha_2=60^\circ$"),
    ("vary_thickness", "z", "thickness_chord", "profile_loss", "thickness_z",
     r"Final KO profile: $\alpha_2=60^\circ$, $s/c=0.7$"),
)


def render(data_dir=None, figure_dir=None, prefix=""):
    data = Path(data_dir) if data_dir is not None else BASE / "data"
    figures = Path(figure_dir) if figure_dir is not None else BASE / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    inputs, outputs = {}, {}
    plt.rcParams.update({"font.size": 11, "axes.spines.right": False})
    for stem, varied, coordinate, ordinate, name, title in SPECS:
        path = data / f"{stem}.csv"
        inputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding="utf-8", newline="") as stream:
            rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
        fig, ax = plt.subplots(figsize=(9, 5.8))
        for value, color in zip(sorted({r[varied] for r in rows}), COLORS):
            curve = [r for r in rows if r[varied] == value]
            label = (rf"$\alpha_2={value:g}^\circ$" if varied == "alpha2_deg"
                     else rf"$s/c={value:g}$" if varied == "pitch_chord" else rf"$z={value:g}$")
            ax.plot([r[coordinate] for r in curve], [r[ordinate] for r in curve],
                    color=color, lw=1.9, label=label)
        if coordinate == "z":
            ax.set_xlim(*summary["z_range"])
            ax.set_xlabel(r"$z=-\theta_{\mathrm{in}}/\alpha_2$")
            ax.set_ylabel(r"Blended reference loss $Y_{p,\mathrm{ref}}=B$")
            for z in (0., 1.):
                ax.axvline(z, color=".6", ls=":", lw=.8)
            top = ax.secondary_xaxis("top")
            top.set_xticks([-1, 0, 1], ["Zero turning", "Axial inlet", "Impulse"])
            top.tick_params(labelsize=9)
            note = ("Reference blend only: before thickness, Reynolds, Mach, shock and KO scaling.\n"
                    "At z = -1: 90% of reaction; negative tail approaches 80% without a flat cutoff.")
        else:
            ax.set_xlim(*summary["thickness_range"])
            ax.set_xlabel(r"Maximum thickness / true chord, $t_{\max}/c$")
            ax.set_ylabel(r"Final KO profile loss $Y_p$")
            ax.axvline(.2, color=".6", ls=":", lw=.8)
            note = ("Re = 500000; inlet/outlet Mach = 0.2/0.5; zero incidence.\n"
                    "For z <= 0, the retained thickness exponent is zero; the angular curves remain distinct.")
        ax.set_ylim(bottom=0.)
        ax.grid(alpha=.16)
        ax.legend(loc="upper left", frameon=False)
        fig.suptitle(title, fontsize=14, y=.98)
        fig.text(.5, .025, note, ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .105, 1, .91))
        for ext in ("png", "pdf"):
            output = figures / f"{prefix}{name}.{ext}"
            fig.savefig(output, dpi=180)
            outputs[output.name] = hashlib.sha256(output.read_bytes()).hexdigest()
        plt.close(fig)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == inputs[path.name]
    result = dict(figures=3, formats=["png", "pdf"], model_calls=0,
                  data_hashes=inputs, figure_hashes=outputs,
                  renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (data / "render_manifest.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    return result
