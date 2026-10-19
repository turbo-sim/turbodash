"""Independent ink readings of KO (1982) Figure 14, with a review overlay.

Guides identify visible curve strokes in the source scan, not model predictions.
Where lettering touches the upper edge, trace the lower stroke edge and subtract
the measured two-pixel half-width. The unresolved origin is not digitized.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[2]
sys.path.insert(0, str(REPO))
from development.loss_model_analysis.pdf_sources import chart_image, chart_pixels_sha256
CROP = (260, 145, 1250, 770)
X_TICKS = np.array([219., 387., 560., 730., 899.])
Y_TICKS = np.array([530.5, 359.5, 197., 36.5])
# Pixel coordinates within the in-memory Figure 14 crop. Letters lie above strokes.
GUIDES = {
    'nozzle': [(310,514),(340,504),(370,491),(400,477),(430,461),
               (460,443),(490,424),(520,405),(550,383),(580,359),
               (610,334),(640,308),(670,284),(700,256),(730,230),
               (760,202),(790,172),(820,144),(850,116),(880,86),(900,67)],
    'impulse': [(310,522),(340,518),(370,510),(400,503),(430,495),
                (460,485),(490,475),(520,464),(550,451),(580,439),
                (610,426),(640,413),(670,401),(700,386),(730,371),
                (760,357),(790,342),(820,326),(850,311),(880,296),(900,285)],
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_source():
    """Validate the recorded source crop in memory; no image files are written."""
    return chart_image("ko_trailing")


def chart_coordinates(x, y):
    # Remove measured scan skew before interpolation through each printed tick.
    # Piecewise calibration also resolves the nonuniform vertical tick spacing.
    xc = x-3*(530.5-y)/(530.5-36.5)
    yc = y-(x-219)/680
    ratio = np.interp(xc, X_TICKS, np.arange(5)*.1)
    loss = np.interp(yc, Y_TICKS[::-1], np.array([0., .05, .1, .15])[::-1])
    return float(ratio), float(loss)


def digitize():
    grey = np.asarray(chart_image("ko_trailing").convert('L'), dtype=float)
    rows = []
    for family, guides in GUIDES.items():
        for index, (x, seed) in enumerate(guides):
            y = np.arange(seed-8, seed+9)
            darkness = 255-np.median(grey[y, x-1:x+2], axis=1)
            ids = np.flatnonzero(darkness > 105)
            groups = np.split(ids, np.flatnonzero(np.diff(ids) > 1)+1)
            groups = [g for g in groups if len(g) >= 2]
            group = min(groups, key=lambda g: abs(np.mean(y[g])-seed))
            if len(group) > 7:
                centre = float(y[group[-1]]-2)
                method, sigma_pixels = 'lower edge; lettering touches stroke', 3.
            else:
                centre = float(np.average(y[group], weights=darkness[group]))
                method, sigma_pixels = 'ink centroid', 2.
            ratio, loss = chart_coordinates(x, centre)
            # Include axis placement and scan distortion, not just stroke width.
            sigma = float(np.hypot(sigma_pixels*.05/162, .00035))
            rows.append(dict(family=family, point=index, pixel_x=x, pixel_y=centre,
                             thickness_opening=ratio, energy_loss=loss,
                             sigma_energy=sigma, sigma_ratio=.0018, method=method,
                             holdout=int(index % 4 == 2)))
    write_csv(BASE/'data/digitized_points.csv', rows)
    write_json(BASE/'data/calibration.json', dict(
        figure=14, printed_page=116, pdf_page=6, render_dpi=300, crop_box=list(CROP),
        x_ticks_pixels=X_TICKS.tolist(), x_ticks_values=(np.arange(5)*.1).tolist(),
        y_ticks_pixels=Y_TICKS.tolist(), y_ticks_values=[0., .05, .1, .15],
        axis_map='Piecewise linear through printed ticks, after measured scan-skew correction',
        vertical_axis_skew_pixels=3, horizontal_axis_skew_pixels=1,
        source_pdf_sha256=sha256(BASE/'source/kacker_okapuu_1982.pdf'),
        source_pixels_sha256=chart_pixels_sha256("ko_trailing"), digitizer_sha256=sha256(__file__),
        points=len(rows), uncertainty='Stroke/axis/scan reading uncertainty; not experimental scatter. '
        'Letter-contact points use the lower edge minus a two-pixel half-width with larger uncertainty.',
        excluded='Merged origin below approximately t/o=0.05. Zero loss at zero thickness is an imposed '
        'physical endpoint, not a fabricated measured point. Existing source table values are not digitization input.'))
    return rows


if __name__ == '__main__':
    print(f'Digitized {len(digitize())} ink points.')
