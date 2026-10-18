"""Reproduce independent, manually guided digitization of AM Figure 4(a).

The point arrays are observed ink centres or explicitly marked approximate
readings in source/nozzle.png, not evaluations of a loss correlation. Calibration
interpolates the printed grid only; it does not add curve samples. No fitted model
is imported by this script.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

# Each entry is a visually assigned curve centre (pixel x, pixel y).  Dashed
# curves are sampled only on visible dashes, never across their gaps.  The
# crowded hatched wedge is handled separately below so its approximate readings
# cannot be confused with the resolved ink centres in this dictionary.
POINTS = {
    40: [(940,680),(981,706),(1100,767),(1140,784),(1180,799),
         (1260,824),(1299,837),(1340,846),(1418,862.5),(1457,869),
         (1496,874.5),(1733,892.5),(1774,894.5),(1814,896),
         (1894,896),(1934,896),(1974,895),(1995,895)],
    50: [(940,667),(981,691.5),(1020,713.5),(1100,755),(1140,771),
         (1180,786),(1260,812.5),(1299,821.5),(1340,833),(1418,850),
         (1457,856.5),(1496,863),(1575,869),(1614,872),(1654,874),
         (1733,875.5),(1774,876),(1894,874.5),(1934,870),(1974,865),
         (1995,862)],
    60: [(940,657.5),(981,680),(1020,702),(1100,742),(1140,759.5),
         (1180,775),(1260,801.5),(1299,811.5),(1340,823),(1418,839),
         (1457,845),(1496,849),(1575,854),(1614,856),(1654,854),
         (1733,848),(1774,841.5),(1814,833),(1894,813),(1934,802),
         (1974,787.5),(1995,780.5)],
    65: [(940,648),(981,667.5),(1020,687),(1100,722.5),(1140,739),
         (1180,751.5),(1260,775.5),(1299,785),(1340,794.5),
         (1418,804.5),(1457,807.5),(1496,810),(1575,809),(1614,806.5),
         (1654,802),(1733,788.5),(1774,779),(1814,769.5),(1894,744),
         (1974,713.5),(1995,705)],
    70: [(785,532.5),(824,561),(863,585),(940,631),(981,653),
         (1020,674),(1100,706),(1140,718),(1260,740.5),(1299,746),
         (1340,750.5),(1418,748.5),(1457,746),(1496,741.5),
         (1614,718.5),(1654,712.5),(1733,693),(1774,681.5),(1814,669),
         (1894,641),(1934,625),(1974,609),(1995,600)],
    75: [(940,608),(981,622.5),(1020,636),(1075,651.5),(1105,658),
         (1150,664.5),(1180,669),(1260,673),(1299,672),(1330,670),
         (1360,669.5),(1418,664),(1465,658),(1496,652),(1540,643),
         (1575,636),(1654,618.5),(1733,597),(1774,587.5)],
    80: [(910,550),(940,556),(955,561),(1045,578),(1100,586),
         (1140,589.5),(1180,591),(1210,591),(1260,588.5),(1285,587),
         (1315,583),(1340,578),(1418,561.5),(1457,552.5),(1480,546),
         (1510,540),(1525,535.5)],
}

# Requested extension towards s/c=.30.  The outer boundaries are visible; the
# interior curves are partly obscured by hatch marks. These are manual estimates
# following the visible strokes and their ordering in that envelope, NOT five
# independently resolved line centres. No high-angle points are invented where
# the 75/80-degree dashes are absent. Larger uncertainty and a distinct quality
# label make these observations identifiable in fitting and review.
APPROXIMATE_CROWDED_POINTS = {
    40: [(746,548),(785,576),(824,601.5),(863,629),(901,656)],
    50: [(746,540),(785,565),(824,592),(863,616),(901,642)],
    60: [(746,533),(785,554),(824,580),(863,606.5),(901,632)],
    65: [(746,519),(785,545),(824,571.5),(863,594),(901,621)],
    70: [(746,507),(901,609)],
}


def all_points():
    """Retain point provenance while merging resolved and approximate readings."""
    for angle, points in POINTS.items():
        selected = [(x,y,False) for x,y in points]
        selected += [(x,y,True) for x,y in APPROXIMATE_CROWDED_POINTS.get(angle,[])]
        yield angle, sorted(selected)

# Printed horizontal lines: rows are x pixel; columns are Yp = 0,.02,...,.12.
# Lines are not perfectly parallel/equally spaced in the scan, so the nearby
# printed cell is used instead of forcing a single global affine transform.
HORIZONTAL_X = [500,800,1000,1200,1400,1600,1800,2000]
HORIZONTAL_Y = [
    [1049,891,729.5,571.5,415,258.5,105.5],
    [1045.33,889,729,570.5,415,258.5,104],
    [1044.5,887,728,569.5,415,256.5,103],
    [1043,886,728,569.5,414,256.5,102.5],
    [1041,885,728,568,413,255,100],
    [1040,885,728,568,409.5,255,99],
    [1038,884.5,728,567,409,255,98.5],
    [1038.5,882.5,726,566,408,254,97.5],
]
# Printed vertical lines: rows are s/c=0,.1,...,1.2, columns y=500,650,800,950.
VERTICAL_Y = [500,650,800,950]
VERTICAL_X = [
    [281,279.86,280,281], [434,435.5,435.5,434],
    [592,592,591,590.5], [745,747,746,747],
    [900,900.5,902.5,902.5], [1058,1060,1061,1061],
    [1218,1219,1219.5,1220], [1377,1378.5,1378.5,1378.5],
    [1533,1535,1535,1535], [1691.5,1691.5,1691.5,1692],
    [1852,1853,1854,1855], [2011.5,2011.5,2013.5,2014],
    [2167,2168,2168,2168],
]


def pixel_to_chart(x, y):
    """Convert an observed pixel using adjacent printed grid lines."""
    vertical_x = [np.interp(y, VERTICAL_Y, values) for values in VERTICAL_X]
    spacing = np.interp(x, vertical_x, np.arange(13) / 10)
    horizontal_y = [np.interp(x, HORIZONTAL_X, values)
                    for values in np.asarray(HORIZONTAL_Y).T]
    yp = np.interp(y, horizontal_y[::-1], (np.arange(7) * .02)[::-1])
    return float(spacing), float(yp)


def main():
    source = ROOT / "source/nozzle.png"
    im = np.asarray(Image.open(source).convert("L"))
    rows = []
    for angle, points in all_points():
        for x, y, approximate in points:
            spacing, yp = pixel_to_chart(x, y)
            quality = "clear_ink"
            sigma = .00050 if angle < 75 else .00065
            if (angle == 40 and x == 1260) or (angle == 50 and x == 1020):
                quality, sigma = "broken_ink_centre", .00075
            if (angle == 65 and x == 1100) or (angle == 80 and x in (1045,1418)):
                quality, sigma = "near_grid_line", .00075
            if angle == 70 and x < 900:
                quality, sigma = "clear_outer_edge_of_hatched_wedge", .00080
            if approximate:
                quality = "approximate crowded ink"
                sigma = (.0015 if x >= 900 else .002 if x >= 860
                         else .0025 if x >= 820 else .003)
                if angle == 70:
                    sigma = .0015
            # A centre may sit in a tiny white hole in a thick printed stroke;
            # require nearby ink so accidental blank-space entries are caught.
            px, py = round(x), round(y)
            if not approximate:
                assert np.min(im[py-3:py+4,px-2:px+3]) < 120, (angle,x,y)
            rows.append(dict(family="nozzle", alpha_deg=angle,
                pitch_chord=spacing, yp=yp, sigma_yp=sigma,
                pixel_x=x, pixel_y=y,
                source_style="dashed" if angle >= 75 else "solid",
                quality=quality))
    path = ROOT / "data/nozzle_points.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    calibration = dict(
        source="Ainley and Mathieson, R&M 2974, Figure 4(a), printed page 24",
        source_image="source/nozzle.png",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_dimensions_px=[im.shape[1],im.shape[0]],
        page_render_width_px=6400,
        crop_box_in_page_25_px=[1050,670,3370,1860],
        conditions=dict(inlet_angle_deg=0, maximum_thickness_chord=.2,
                        reynolds=200000, mach_less_than=.6),
        method="Manually identify each curve using its printed angle label and trace its ink. Record resolved pixel centres and separately marked approximate readings through the hatched wedge at s/c~.30-.40. No Aungier values or fitted curves are consulted.",
        calibration="Local printed-grid rectification: interpolate grid-line positions at the sample pixel, then scale between adjacent grid lines. This calibrates axes only and never interpolates missing curve points.",
        horizontal_x_pixels=HORIZONTAL_X,
        horizontal_y_pixels=HORIZONTAL_Y,
        horizontal_yp_values=(np.arange(7)*.02).tolist(),
        vertical_y_pixels=VERTICAL_Y,
        vertical_x_pixels=VERTICAL_X,
        vertical_pitch_chord_values=(np.arange(13)/10).tolist(),
        calibration_notes=[
            "Grid-line centres measured from dark pixels. Yp=.02 grid at x=1600,1800,2000 was identified manually to exclude the nearby 40/50-degree traces.",
            "Yp=.06 grid at x=800 was obscured by the hatched wedge; its local position was read from the continuation either side.",
            "Vertical s/c=.6 grid at y=800 was locally obscured; its grid-line position was taken between the unambiguous segments.",
        ],
        uncertainty="sigma_yp is a heuristic reading scale, not a statistical standard deviation or experimental uncertainty. Clear solid ink: .00050; dashed: .00065; broken ink/near grid: .00075; wedge outer edge: .00080. Approximate crowded ink: .0015-.003, covering uncertain curve identification within the hatched envelope. Approximate readings are not independently resolved ink centres. Resolved-point scales cover line thickness, about 2-pixel grid placement and local ambiguity. Pixel x reading uncertainty is approximately 2 pixels (~.0013 s/c).",
        approximate_readings={
            "quality_label":"approximate crowded ink",
            "count":sum(len(p) for p in APPROXIMATE_CROWDED_POINTS.values()),
            "spacing_range":"approximately .30-.40",
            "angles_deg":[40,50,60,65,70],
            "rationale":"User requested extension to s/c=.30 and accepted approximate digitization. The visible outer wedge and fragments of the internal traces guide these estimates; labels are followed from the resolved curves on the right. These points carry .0015-.003 reading scales and must not be presented as independently resolved measurements.",
        },
        omissions=[
            "Interior 40-65-degree curves below s/c~.40 are partly unresolved in a hatched/merged wedge. Their explicitly approximate readings replace the earlier omission and have substantially larger uncertainty.",
            "40-degree line omitted near s/c~.83-.88 where it overlaps the Yp=.02 grid.",
            "Samples at solid-curve/grid intersections omitted when the individual line centre cannot be read.",
            "Dashed curves sampled only on visible dashes: 75-degree trace stops before its label near s/c~.95 and 80-degree trace near s/c~.80. No continuation is invented.",
        ],
        number_of_points=len(rows),
        points_per_angle={str(a):len(p) for a,p in all_points()},
    )
    (ROOT / "data/nozzle_calibration.json").write_text(
        json.dumps(calibration,indent=2)+"\n",encoding="utf-8")
    fig, ax = plt.subplots(figsize=(15.5,8.5))
    ax.imshow(im,cmap="gray",vmin=0,vmax=255)
    colors = plt.get_cmap("tab10")
    for i,(angle,points) in enumerate(all_points()):
        for approximate,marker in [(False,"o"),(True,"s")]:
            xy=np.asarray([(x,y) for x,y,is_approx in points if is_approx == approximate])
            if not len(xy):
                continue
            ax.scatter(xy[:,0],xy[:,1],s=32,facecolors="none",edgecolors=colors(i),
                       marker=marker,linewidths=1.1,
                       label=f"{angle}\N{DEGREE SIGN} ({len(points)} samples)" if not approximate else None)
    ax.set_xlim(250,2180)
    ax.set_ylim(1070,420)
    ax.set_xlabel("Pixel x in source/nozzle.png")
    ax.set_ylabel("Pixel y in source/nozzle.png")
    ax.set_title("Ainley-Mathieson nozzle chart: manually read source traces\n"
                 "Circles: resolved ink; squares: approximate crowded readings with larger uncertainty")
    ax.legend(ncol=4,loc="upper center",bbox_to_anchor=(.5,-.11),frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "figures/nozzle_digitization.png",dpi=160,bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {len(rows)} nozzle samples, including "
          f"{calibration['approximate_readings']['count']} approximate crowded readings.")


if __name__ == "__main__":
    main()
