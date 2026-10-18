"""Reproduce manually assigned ink-centre readings of KO (1982), Figure 6.

Coordinates refer to source/figure_6.png. They were read from the supplied
paper, independently of the existing TurboDash table or any fitted model.
Points obscured by labels or merged with the horizontal axis are excluded.
"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
X_TICKS = np.array([373., 544., 717., 889., 1063., 1237.])
R_TICKS = np.array([.5, .6, .7, .8, .9, 1.])
Y_TICKS = np.array([594., 507., 421., 333., 246., 159., 72.])
F_TICKS = np.arange(1., 2.21, .2)
POINTS = {
    "stator": [(386,426.5),(420,446),(454,465.5),(488,484.5),
               (522,501.5),(556,518),(590,532.5),(624,545.5),
               (658,557),(692,568),(726,577.5),(760,585)],
    "rotor": [(386,98.5),(420,140.5),(454,181),(488,221.5),
              (522,260.5),(556,295),(590,330),(624,361.5),
              (658,391.5),(692,420),(726,446),(760,470),
              (794,492),(828,512),(862,528.5),(896,544),
              (930,556),(964,566),(998,576),(1032,584)],
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digitize():
    x_map = np.polyfit(X_TICKS, R_TICKS, 1)
    y_map = np.polyfit(F_TICKS, Y_TICKS, 1)
    rows = []
    for family, points in POINTS.items():
        for x, y in points:
            # The printed baseline moves down four image pixels toward the right.
            # Correct that measured scan skew before applying the y scale.
            y_corrected = y-4*(x-210)/(1237-210)
            factor = (y_corrected-y_map[1])/y_map[0]
            rows.append(dict(family=family, pixel_x=x, pixel_y=y,
                             hub_tip_ratio=float(np.polyval(x_map, x)),
                             hub_mach_factor=float(factor), sigma_factor=.006))
    data = BASE/"data"
    data.mkdir(exist_ok=True)
    with (data/"digitized_points.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    calibration=dict(figure=6,printed_page=113,pdf_page=3,
        source_pdf_sha256=sha256(BASE/"source/kacker_okapuu_1982.pdf"),
        source_image_sha256=sha256(BASE/"source/figure_6.png"),
        render_dpi=400,crop_box=[1731,185,3089,1015],
        x_ticks_pixels=X_TICKS.tolist(),x_ticks_values=R_TICKS.tolist(),
        y_ticks_pixels=Y_TICKS.tolist(),y_ticks_values=F_TICKS.tolist(),
        x_map=x_map.tolist(),y_map=y_map.tolist(),baseline_skew_pixels=4,
        uncertainty="Assigned sigma=0.006 in f_hub (about 2.6 pixels); tracing/scan uncertainty, not experimental error bars.",
        excluded="No points invented below r=0.5 or in the unresolved axis-merging tails. Nozzle means stator.",
        points=len(rows),digitizer_sha256=sha256(__file__))
    (data/"calibration.json").write_text(json.dumps(calibration,indent=2)+"\n",encoding="utf-8")
    return rows


if __name__=="__main__":
    print(f"Digitized {len(digitize())} independently read curve points.")
