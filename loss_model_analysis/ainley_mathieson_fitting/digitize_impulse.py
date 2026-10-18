"""Digitize independently selected ink in Ainley--Mathieson Figure 4(b).

The manually selected pixel windows below come from the original scan, not
from a correlation. This script snaps each window to its observed ink, calibrates
against the printed grid, and saves measured coordinates and a review overlay.
The crowded left edge additionally has explicitly approximate readings with
larger uncertainty. Where the 55-degree trace is merged, its position is read
inside the visible 50--60-degree wedge in label order; it is not an independently
resolved centreline. These approximate readings are never presented as precise
ink measurements. No loss correlation is used to choose any reading.
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
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source" / "arc-rm-2974.pdf"
sys.path.insert(0, str(ROOT.parents[1]))
from loss_model_analysis.pdf_sources import chart_image, chart_pixels_sha256

# (pixel x, approximate ink-centre y, optional half-height of sampling window).
# Several curves contain white centres in thick strokes. The centroid includes
# both ink edges. Omitted positions include dash gaps and ambiguous grid joins.
# At the left, the 55-degree trace is not separable until approximately s/c=.4.
GUIDES = {
    40: [(782,638,13),(821,707,12),(859,766,12),(938,868),(977,911),
         (1017,950),(1097,1017),(1179,1068),(1259,1108),(1298,1122),
         (1337,1134),(1416,1147),(1455,1149),(1495,1149),(1574,1143),
         (1614,1138),(1653,1132),(1733,1116),(1773,1106),(1813,1093),
         (1840,1083)],
    50: [(782,612,12),(821,674,11),(938,833),(1017,914,11),(1097,976),
         (1138,1004),(1179,1027),(1259,1059),(1298,1069),(1337,1078),
         (1416,1087),(1455,1086),(1495,1082),(1574,1068),(1614,1059),
         (1733,1021),(1773,1005),(1813,989),(1840,977)],
    55: [(938,798),(977,835),(1017,867),(1097,923),(1138,945),
         (1179,962),(1259,985),(1298,992),(1337,997),(1416,1000),
         (1455,997),(1495,991),(1574,971),(1614,959),(1653,947),
         (1733,915),(1773,897),(1840,863)],
    60: [(798,601,12),(821,637,11),(859,689,11),(938,772),(977,800),
         (1017,821),(1097,849),(1138,858),(1179,863),(1259,866),
         (1298,865),(1337,861),(1416,847),(1455,839),(1495,829),
         (1574,805),(1614,790),(1653,775),(1733,744),(1813,709),
         (1840,697)],
    65: [(782,503,10),(815,547),(859,606,10),(938,673),(977,699),
         (1011,714,7),(1105,746),(1130,750),(1171,757),(1251,754),
         (1298,744),(1416,706),(1455,690),(1487,678),(1574,637),
         (1614,616),(1653,596),(1733,549),(1773,524),(1813,497),
         (1840,482)],
    70: [(782,454),(821,495),(859,526),(977,584),(1029,596),
         (1089,600),(1130,596),(1179,588),(1259,557),(1298,540),
         (1337,520),(1408,481),(1455,453),(1487,430),(1574,372),
         (1614,343),(1653,316),(1773,226),(1813,197),(1840,175)],
}

# Explicit approximate readings requested for the crowded s/c=.30--.40 region.
# Each tuple is (nominal pitch/chord, visually read pixel y, sigma_yp, reason).
# The x pixel coordinate is obtained from that nominal chart coordinate using
# the calibrated printed grid. These y positions are manual image readings,
# not centres snapped to a grid line or values from Aungier/a fitted surface.
# The unresolved 55-degree trace is interpreted within the visible wedge in
# increasing-angle order. Its uncertainty spans much of that wedge.
APPROXIMATE_READINGS = {
    40: [(.300,566.,.0025,"curve start obscured by vertical grid"),
         (.400,821.,.0020,"curve crossing vertical grid")],
    50: [(.300,542.,.0025,"curve start obscured by vertical grid"),
         (.375,736.,.0025,"curve overlaps horizontal grid"),
         (.400,783.,.0020,"curve crossing vertical grid")],
    55: [(.300,526.,.0030,"unresolved trace inside 50--60 degree wedge"),
         (.325,596.,.0030,"unresolved trace inside 50--60 degree wedge"),
         (.350,656.,.0025,"unresolved trace inside 50--60 degree wedge"),
         (.375,712.,.0025,"unresolved trace inside 50--60 degree wedge"),
         (.400,755.,.0020,"separating trace obscured by vertical grid")],
    60: [(.300,512.,.0025,"curve start obscured by vertical grid"),
         (.325,580.,.0025,"curve overlaps horizontal grid"),
         (.400,734.,.0020,"curve crosses two grid lines")],
    65: [(.300,442.,.0025,"dashed curve start obscured by vertical grid"),
         (.350,562.,.0020,"dash approaches horizontal grid"),
         (.400,644.,.0020,"dash crossing vertical grid")],
    70: [(.300,402.,.0025,"dashed curve start overlaps grid intersection"),
         (.400,550.,.0020,"dash crossing vertical grid")],
}

# Printed .1-pitch/chord and .02-loss grid lines, in source-crop pixels.
X_SEEDS = [281,435,589,744,898,1057,1220,1376,1534,1693,1854,2014,2168]
Y_SEEDS = [1668,1517,1357,1204,1042,883,727,570,414,257,102]


def fit_grid_line(a, seed, vertical):
    """Fit a slightly skewed printed grid line using many clear local patches."""
    coordinates = np.arange(160,1630,31) if vertical else np.arange(335,2130,37)
    centres = []
    for q in coordinates:
        if vertical:
            patch = a[q-6:q+7,seed-12:seed+13]
            strength = (patch < 150).mean(axis=0)
        else:
            patch = a[seed-12:seed+13,q-6:q+7]
            strength = (patch < 150).mean(axis=1)
        ids = np.flatnonzero(strength >= max(.65, .80*strength.max()))
        if len(ids):
            centre = np.average(ids+seed-12, weights=strength[ids])
            centres.append((float(q),float(centre)))
    q,p = np.asarray(centres).T
    use = np.ones(len(q),bool)
    for _ in range(4):
        coeff = np.polynomial.polynomial.polyfit(q[use],p[use],1)
        residual = p-np.polynomial.polynomial.polyval(q,coeff)
        use = np.abs(residual) < max(2.5,3*np.median(np.abs(residual)))
    return {"intercept":float(coeff[0]),"slope":float(coeff[1]),
            "rms_px":float(np.sqrt(np.mean(residual[use]**2))),
            "n_patches":int(use.sum()),
            "retained_patch_centres_px":np.column_stack([q[use],p[use]]).tolist()}


def physical(x,y,vertical,horizontal):
    gx = [v["intercept"]+v["slope"]*y for v in vertical]
    gy = [v["intercept"]+v["slope"]*x for v in horizontal]
    pitch = np.interp(x,gx,np.arange(13)/10)
    loss = np.interp(y,gy[::-1],(np.arange(11)*.02)[::-1])
    return float(pitch),float(loss)


def main():
    a = np.asarray(chart_image("am_impulse").convert("L"))
    vertical = [fit_grid_line(a,s,True) for s in X_SEEDS]
    horizontal = [fit_grid_line(a,s,False) for s in Y_SEEDS]
    rows,windows = [],[]
    for angle,guides in GUIDES.items():
        for guide in guides:
            x,y = guide[:2]
            radius = guide[2] if len(guide)==3 else 9
            yy = np.arange(y-radius,y+radius+1)
            patch = a[y-radius:y+radius+1,x-1:x+2]
            # Darkness weights suppress antialiasing; the three-column strip
            # gives a centreline measurement rather than a selected dark edge.
            weights = np.where(patch<170,255.-patch,0.).sum(axis=1)
            if np.count_nonzero(weights)<3:
                raise ValueError(f"Insufficient ink at {angle}, {guide}")
            measured_y = float(np.average(yy,weights=weights))
            spread = float(np.sqrt(np.average((yy-measured_y)**2,weights=weights)))
            sig_px = max(2.8,np.hypot(spread/1.5,2.0))
            s,yp = physical(float(x),measured_y,vertical,horizontal)
            _,yp_hi = physical(float(x),measured_y-sig_px,vertical,horizontal)
            row = dict(family="impulse",alpha_deg=angle,pitch_chord=s,yp=yp,
                       sigma_yp=abs(yp_hi-yp),pixel_x=float(x),pixel_y=measured_y,
                       source_style="dashed" if angle>=65 else "solid",
                       quality="clear_selected_ink")
            rows.append(row)
            windows.append(dict(alpha_deg=angle,pixel_x=x,guide_y=y,
                                half_height_px=radius,measured_y=measured_y,
                                ink_spread_px=spread))
    approximate_records = []
    for angle,readings in APPROXIMATE_READINGS.items():
        for s,y,sigma,reason in readings:
            gx = [v["intercept"]+v["slope"]*y for v in vertical]
            x = float(np.interp(s,np.arange(13)/10,gx))
            _,yp = physical(x,y,vertical,horizontal)
            rows.append(dict(family="impulse",alpha_deg=angle,pitch_chord=s,
                             yp=yp,sigma_yp=sigma,pixel_x=x,pixel_y=y,
                             source_style="dashed" if angle>=65 else "solid",
                             quality="approximate crowded ink"))
            approximate_records.append(dict(alpha_deg=angle,pitch_chord=s,
                                             pixel_x=x,pixel_y=y,sigma_yp=sigma,
                                             reason=reason))
    rows.sort(key=lambda row:(row["alpha_deg"],row["pitch_chord"]))
    out = ROOT/"data"/"impulse_points.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    calibration = {
        "source":"Ainley and Mathieson, ARC R&M 2974, Figure 4(b), printed page 24",
        "pdf_page_index":25,
        "source_pdf":"source/arc-rm-2974.pdf",
        "source_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "source_pixels_sha256":chart_pixels_sha256("am_impulse"),
        "crop_from_6400_wide_page_px":[1050,2240,3370,4080],
        "pixel_origin":"upper left of the in-memory Figure 4(b) crop; x rightward, y downward",
        "calibration_method":"Local bilinear grid calibration: each printed grid line is fitted as a straight line; measured ink is mapped between its neighboring grid lines. This corrects scan skew and uneven grid spacing, and never interpolates curve data.",
        "vertical_lines_pitch_chord":list(np.arange(13)/10),
        "horizontal_lines_yp":list(np.arange(11)*.02),
        "vertical_line_fits":vertical,"horizontal_line_fits":horizontal,
        "manual_windows":windows,
        "approximate_crowded_readings":approximate_records,
        "independence":"All curve guides selected from original figure ink without evaluating Aungier or proposed fits.",
        "uncertainty":"sigma_yp is a heuristic plotting uncertainty, not a statistical experimental error. Clear readings combine source-line thickness/ambiguity with a 2-pixel grid allowance and minimum 2.8 pixels. Approximate crowded readings use explicit sigma_yp=.002--.003; the unresolved 55-degree trace is interpreted within the visible 50--60-degree wedge in label order, not independently resolved. Original chart rounding and experimental uncertainty are unknown.",
        "omissions":["Dash gaps outside the explicitly approximate low-spacing region are not filled.","Ambiguous grid crossings outside the explicitly approximate low-spacing region are omitted.","The 55-degree left trace below about s/c=0.4 is unresolved: the added wedge readings are approximate guidance, not independent precision measurements.","The s/c=1.0 endpoint is a vertical grid line; the closest clear ink is measured instead.","Small isolated scan specks are ignored using manually selected curve windows."],
        "conditions":{"zero_incidence":True,"max_thickness_chord":.2,
                      "Re":200000,"Mach_less_than":.6},
        "point_counts":{str(k):sum(r["alpha_deg"]==k for r in rows) for k in GUIDES},
        "quality_counts":{q:sum(r["quality"]==q for r in rows)
                          for q in ("clear_selected_ink","approximate crowded ink")},
    }
    (ROOT/"data"/"impulse_calibration.json").write_text(json.dumps(calibration,indent=2)+"\n",encoding="utf-8")
    fig,ax=plt.subplots(figsize=(14,11))
    ax.imshow(a,cmap="gray",vmin=0,vmax=255)
    colors=plt.get_cmap("tab10").colors
    for j,angle in enumerate(GUIDES):
        rr=[r for r in rows if r["alpha_deg"]==angle]
        clear=[r for r in rr if r["quality"]=="clear_selected_ink"]
        approx=[r for r in rr if r["quality"]=="approximate crowded ink"]
        ax.scatter([r["pixel_x"] for r in clear],[r["pixel_y"] for r in clear],
                   s=35,facecolors="none",edgecolors=colors[j],linewidths=1.1,
                   label=f"{angle}\N{DEGREE SIGN} ({len(rr)} readings)")
        ax.scatter([r["pixel_x"] for r in approx],[r["pixel_y"] for r in approx],
                   s=43,marker="^",facecolors="none",edgecolors=colors[j],
                   linewidths=1.1)
    ax.set_title("Original Ainley--Mathieson impulse chart\nCircles: clear ink; triangles: approximate crowded readings (larger uncertainty)")
    ax.set_xlabel("Source-crop pixel x");ax.set_ylabel("Source-crop pixel y")
    ax.legend(loc="lower right",fontsize=9,framealpha=.95)
    fig.tight_layout()
    (ROOT/"figures").mkdir(exist_ok=True)
    fig.savefig(ROOT/"figures"/"impulse_digitization.pdf",dpi=170)
    plt.close(fig)
    print(json.dumps({"points":len(rows),"counts":calibration["point_counts"],
                      "csv":str(out),"max_grid_line_rms_px":max(v["rms_px"] for v in vertical+horizontal)}))


if __name__=="__main__":
    main()
