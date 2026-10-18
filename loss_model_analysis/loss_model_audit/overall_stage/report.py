"""Individual and side-by-side sensitivity reports.

Reuse saved figures: python -B report.py --comparison-only
"""
from collections import Counter
from html import escape
from html.parser import HTMLParser
import json
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from study import (BASE,DATA,MODELS,PARAMETERS,SWEEPS,FIXED,
                   read_samples,write_json,sha256,source_hashes,model_directory,CONFIG_KEYS)

INK="#172c43"; MUTED="#596d7f"
PURPOSES={
 "stator_exit_angle":"Test stator turning, rotor inlet triangles and unrecovered exit kinetic energy while redesigning the metal angles at every point.",
 "pitch_chord":"Search for the spacing optimum, including continuation outside the Ainley-Mathieson chart, with the same pitch/true-chord ratio in both rows.",
 "aspect_ratio":"Test secondary/clearance benefits against the Reynolds change caused by changing true chord, with equal height/chord ratios in both rows.",
 "trailing_ratio":"Check that thickening both trailing edges increases loss and reduces efficiency, including the zero-thickness endpoint.",
 "specific_speed":"Expose the coupled response of rotational speed, annulus shape, chord, Reynolds correction and hub shock at fixed mass flow.",
 "mass_flow":"Isolate size at fixed dimensionless design. Larger chord Reynolds numbers approach the finite Reynolds-correction limit.",
 "isentropic_mach":"Change expansion pressure ratio and inspect the combined compressibility, shock and Reynolds response."
}
ISENTROPIC_PURPOSES={
 "stator_exit_angle":"Isolate how stator exit angle changes the velocity triangles and unrecovered exit kinetic energy, with every row-loss component set to zero.",
 "pitch_chord":"Verify that spacing has no aerodynamic efficiency effect when the loss model is isentropic; retain the common geometry checks.",
 "aspect_ratio":"Verify that blade aspect ratio affects sizing but cannot create secondary or clearance losses in the zero-loss model.",
 "trailing_ratio":"Verify that trailing-edge thickness adds no loss in the isentropic model; retain the same reference-geometry checks as the empirical studies.",
 "specific_speed":"Check that rotational speed and annulus sizing leave the prescribed dimensionless triangles and zero-loss efficiencies unchanged.",
 "mass_flow":"Check size invariance: mass flow changes dimensions and Reynolds numbers, but neither row losses nor normalized velocity triangles.",
 "isentropic_mach":"Vary dimensional expansion and thermodynamic states while checking that the prescribed normalized exit kinetic energy, and hence ideal efficiency, is unchanged."
}

def style():
 plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10.5,"mathtext.fontset":"dejavusans",
  "text.color":INK,"axes.labelcolor":INK,"xtick.color":MUTED,"ytick.color":MUTED,
  "axes.edgecolor":"#c5cfd7","axes.spines.top":False,"axes.spines.right":False,
  "axes.facecolor":"white","figure.facecolor":"white","grid.color":"#dce3e9",
  "grid.alpha":.65,"grid.linewidth":.7,"pdf.fonttype":42,"savefig.facecolor":"white"})

def palette(name):
 return LinearSegmentedColormap.from_list(name+"_readable",plt.get_cmap(name)(np.linspace(.4,.96,256)))

def group_rows(rows,sweep,parameter):
 return [r for r in rows if r["sweep"]==sweep and r["parameter"]==parameter]

def title(sweep):
 return r"Blade velocity ratio at $R=0$" if sweep=="speed_r0" else (r"Blade velocity ratio at $R=0.5$" if sweep=="speed_r05" else r"Reaction at $U/c_0=0.6$")

def filename(sweep,parameter):
 return f"{sweep}_{parameter}"

def plot_group(model,sweep,parameter,rows,ko_rows=None):
 settings=PARAMETERS[parameter];levels=settings["values"]
 norm=(LogNorm if settings.get("log") else Normalize)(levels[0],levels[-1])
 blue,orange=palette("Blues"),palette("Oranges")
 comparison=ko_rows is not None
 fig=plt.figure(figsize=(11.6,7.5));ax=fig.add_axes([.085,.32,.87,.49])
 fig.text(.085,.95,settings["name"],size=21,weight="bold",va="top")
 fixed=r"$R=0$" if sweep=="speed_r0" else (r"$R=0.5$" if sweep=="speed_r05" else r"$U/c_0=0.6$")
 name="Benner with KO comparison" if comparison else MODELS[model]
 fig.text(.085,.893,f"{name}  /  {fixed}",size=12,color=MUTED)
 fig.text(.955,.893,"50 bar / 600 K total inlet",ha="right",size=10,color=MUTED)
 coordinates=sorted({r["coordinate"] for r in rows});coverage=np.zeros(len(coordinates),dtype=int)
 for level in levels:
  bold=np.isclose(level,settings["baseline"],rtol=0,atol=1e-12)
  datasets=[(ko_rows,(0,(4,3)),1.55 if bold else 1.05,2),(rows,"-",2.65 if bold else 1.65,4)] if comparison else [(rows,"-",2.65 if bold else 1.65,4)]
  for dataset,linestyle,width,zorder in datasets:
   curve=sorted([r for r in dataset if r["parameter_value"]==level],key=lambda r:r["coordinate"])
   good=np.array([r["status"]=="ok" for r in curve])
   if dataset is rows:coverage+=good.astype(int)
   x=np.array([r["coordinate"] for r in curve])
   for key,cmap in (("eta_tt",blue),("eta_ts",orange)):
    values=np.array([100*r.get(key,np.nan) if valid else np.nan for r,valid in zip(curve,good)])
    ax.plot(x,values,color=cmap(norm(level)),ls=linestyle,lw=width,zorder=zorder,
            clip_on=model!="isentropic",
            solid_capstyle="round",dash_capstyle="round")
 ax.fill_between(coordinates,0,100,where=coverage==0,color="#edf0f3",zorder=0)
 if sweep=="reaction":
  ax.set(xlim=(-.1,1.1),xlabel=r"Degree of reaction $R$")
  ax.axvspan(-.1,0,color="#e9c98d",alpha=.16);ax.axvspan(1,1.1,color="#e9c98d",alpha=.16)
 else:ax.set(xlim=(0,1.2),xlabel=r"Blade velocity ratio $U/c_0$")
 ax.set(ylim=(0,100),ylabel="Efficiency [%]")
 ax.xaxis.set_major_locator(MultipleLocator(.2));ax.yaxis.set_major_locator(MultipleLocator(20));ax.grid(True)
 ax.tick_params(length=0,pad=7)
 if model=="isentropic":
  note=r"All $\eta_{tt}$ levels coincide at 100%."
  if parameter!="stator_exit_angle":note+="\n"+r"All $\eta_{ts}$ parameter levels also coincide."
  fig.text(.085,.855,note,va="top",fontsize=8.5,color=MUTED)
 for left,cmap,label in ((.085,blue,r"Total-total $\eta_{tt}$"),(.55,orange,r"Total-static $\eta_{ts}$")):
  cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=fig.add_axes([left,.145,.405,.017]),orientation="horizontal")
  cb.set_ticks(levels,labels=[f"{v:g}" for v in levels]);cb.ax.minorticks_off();cb.outline.set_visible(False)
  cb.ax.tick_params(labelsize=8.5,length=3)
  fig.text(left,.191,label,size=10.5,weight="bold",color=cmap(.75))
 if comparison:
  fig.legend(handles=[Line2D([],[],color=INK,lw=2,label="Benner"),Line2D([],[],color=INK,lw=1.2,ls=(0,(4,3)),label="KO")],loc="upper center",bbox_to_anchor=(.52,.25),frameon=False,ncol=2,fontsize=9)
 label=settings["symbol"]+(f" [{settings['unit']}]" if settings['unit'] else "")
 fig.text(.52,.075,f"Curve parameter: {label}   |   bold: baseline {settings['baseline']:g}",ha="center",size=9.5,color=MUTED)
 finite=sum(r["status"]=="ok" for r in rows)
 fig.text(.085,.025,f"{finite:,}/{len(rows):,} finite samples; signed efficiencies retained. Gray: no real reference design.",size=8.4,color=MUTED)
 output={};directory=model_directory(model)/"figures";directory.mkdir(parents=True,exist_ok=True)
 for ext in ("pdf",):
  path=directory/f"{filename(sweep,parameter)}.{ext}";fig.savefig(path,dpi=185)
  output[path.relative_to(BASE).as_posix()]=sha256(path)
 plt.close(fig);return output

def assessment(sweep,parameter,rows):
 good=[r for r in rows if r["status"]=="ok"]
 if not good:return "**Problematic.** No finite reference design exists in this family.",{}
 anchor=.5 if sweep=="reaction" else .6
 by_x={}
 for r in good:by_x.setdefault(r["coordinate"],[]).append(r)
 x=min(by_x,key=lambda q:(-len(by_x[q]),abs(q-anchor)))
 section=sorted(by_x[x],key=lambda r:r["parameter_value"]);first,last=section[0],section[-1]
 axis="$R$" if sweep=="reaction" else "$U/c_0$"
 if good[0]['model']=='isentropic':
  spread=max(max(r['eta_ts'] for r in curve)-min(r['eta_ts'] for r in curve) for curve in by_x.values())
  identity_error=max(abs(r['eta_ts']-(1-r['leaving_energy']/r['isentropic_drop'])) for r in good)
  inactive=parameter!='stator_exit_angle'
  okay=all(r['eta_tt']==1. for r in good) and identity_error<3e-14 and (not inactive or spread<3e-14)
  verdict="Consistent" if okay else "Problematic"
  comment=("The parameter levels overlap because this parameter does not change the prescribed dimensionless triangles and no empirical loss correction is active."
           if inactive else "Stator angle changes exit kinetic energy and therefore total-static efficiency; total-total efficiency remains exactly 100% for every finite reference design.")
  negative=sum(r.get('negative_hub',0)>0 for r in good)
  text=(f"**{verdict}.** All row-loss components are exactly zero and total-total efficiency is exactly 100%. "
        f"{comment} At {axis}={x:g}, total-static efficiency ranges from {100*min(r['eta_ts'] for r in section):.2f}% to {100*max(r['eta_ts'] for r in section):.2f}%. "
        f"Retained finite samples include {negative:,} negative-hub designs. Gaps use the same reference-design checks as the empirical studies.")
  return text,dict(verdict=verdict,comparison_coordinate=x,negative_hub=negative,
                   eta_tt_exactly_one=all(r['eta_tt']==1. for r in good),
                   maximum_parameter_eta_ts_spread=spread,maximum_exit_energy_identity_error=identity_error)
 sentence=(f"At {axis}={x:g}, increasing {PARAMETERS[parameter]['symbol']} from {first['parameter_value']:g} to {last['parameter_value']:g} "
  f"changes total-total efficiency from {100*first['eta_tt']:.1f}% to {100*last['eta_tt']:.1f}% and total-static efficiency from {100*first['eta_ts']:.1f}% to {100*last['eta_ts']:.1f}%.")
 verdict="Review"
 if parameter=="trailing_ratio":
  pairs={}
  for r in good:pairs.setdefault(r['coordinate'],[]).append(r)
  violations=sum(b['eta_tt']>a['eta_tt']+1e-6 for rr in pairs.values() for a,b in zip(sorted(rr,key=lambda r:r['parameter_value']),sorted(rr,key=lambda r:r['parameter_value'])[1:]))
  verdict="Consistent" if violations==0 else "Review"
  comment=f"There are {violations} reversals of the expected thickness trend across adjacent finite parameter levels."
 elif parameter=="pitch_chord":
  best=max(section,key=lambda r:r['eta_tt']);inside=first['parameter_value']<best['parameter_value']<last['parameter_value']
  verdict="Consistent" if inside else "Review"
  comment=f"The best sampled spacing here is {best['parameter_value']:g}; "+("an interior minimum in loss is plausible." if inside else "the boundary optimum needs a wider design check.")
 elif parameter=="specific_speed":
  comment="This is a coupled sizing effect. The positive hub/tip coordinate and finite Reynolds-correction limit bound the empirical inputs while signed hub radii remain in the meanline geometry."
 elif parameter=="isentropic_mach":
  comment="Supersonic trends require review: the extra outlet-supersonic profile multiplier is disabled, so a numerical optimum does not establish aerodynamic validity."
 elif parameter=="mass_flow":
  verdict="Consistent" if last['eta_tt']>=first['eta_tt'] else "Review"
  comment="Chord Reynolds number grows with the square root of mass flow; its correction approaches 0.457305, giving a finite size benefit."
 elif parameter=="aspect_ratio":
  comment="Changing aspect ratio changes secondary/clearance losses and chord Reynolds number together; their balance should be assessed through the saved row components."
 else:
  verdict="Consistent"
  comment="The total-total and total-static responses reflect turning, row losses and leaving kinetic energy."
 if sweep=="reaction":
  tails={}
  for r in good:
   if r['coordinate']>=.9:tails.setdefault(r['parameter_value'],[]).append(r)
  rises=sum(b['eta_tt']>a['eta_tt']+.001 for curve in tails.values()
            for a,b in zip(sorted(curve,key=lambda r:r['coordinate']),sorted(curve,key=lambda r:r['coordinate'])[1:]))
  comment+=(f" Above R=0.9, there are {rises} adjacent total-total efficiency rises larger than 0.1 percentage point. "
             "Shared clearance and KO secondary loading preserve physical turning, with only the mean secant continued from sec80 toward sec87.5.")
  if rises:verdict="Review"
  if rows[0]['model']=='benner':
   comment+=" Benner retains its own secondary correlation's 85-degree flow-angle clips and its separate penetration angular convention; remaining knees require interpretation."
 negative=sum(r.get('negative_hub',0)>0 for r in good);negative_eta=sum(r['eta_tt']<0 for r in good)
 text=f"**{verdict}.** {sentence} {comment} Retained samples include {negative:,} negative-hub designs and {negative_eta:,} negative total-total efficiencies; the latter fall below the displayed axes."
 return text,dict(verdict=verdict,comparison_coordinate=x,negative_hub=negative,negative_eta_tt=negative_eta)

def model_report(model,rows,ko_rows=None):
 pairs=[(s,p) for p in PARAMETERS for s in SWEEPS]
 content=[f"# {MODELS[model]}: single-stage air turbine","","[Common method](../readme.md)","",
  "Blue and orange encode total-total and total-static efficiency. Curves use the common baseline except for the indicated parameter. "
  "Negative hub radii and negative efficiencies remain in the numerical data. Efficiency axes show 0-100%. No hub-ratio boundary markers are drawn.",""]
 if ko_rows is not None:content += ["Benner is shown with solid main curves; matching KO curves use finer dashed lines and the same colors. Both models use the current shock and Reynolds helpers. Benner's secondary correlation and penetration weighting account for the remaining model differences.",""]
 if model=="isentropic":
  content += [r"This study uses the existing production `isentropic` option. Profile, incidence, secondary, trailing-edge and clearance losses are all exactly zero. Geometry and inlet conditions match the empirical studies; the loss functions are bypassed.","",
   "$$",r"\eta_{tt}=1,\qquad\eta_{ts}=1-\frac{v_4^2/2}{h_{01}-h_{4s}}.","$$","",
   r"Total-static efficiency can therefore be below 100% despite zero irreversible loss: it includes unrecovered absolute exit kinetic energy. Curves for pitch/chord, aspect ratio, trailing-edge thickness, specific speed and mass flow coincide because these inputs only change geometry or inactive loss corrections. Expansion Mach changes dimensional states and work, while the prescribed normalized triangles leave this ideal efficiency unchanged. Overlap is expected and is checked numerically.","",
   r"The blue total-total curves lie exactly on the upper 100% boundary. Both efficiencies use solid lines and their existing colormaps; no visual offsets are introduced to separate identical solutions. Zero blade speed and nonreal reference triangles remain explicit gaps, as in the other studies.",""]
 results={}
 for sweep,parameter in pairs:
  selected=group_rows(rows,sweep,parameter);text,metrics=assessment(sweep,parameter,selected)
  key=f"{sweep}/{parameter}";results[key]=metrics
  purpose=(ISENTROPIC_PURPOSES if model=="isentropic" else PURPOSES)[parameter]
  content += [f"## {PARAMETERS[parameter]['name']}: {title(sweep)}","",f"**Purpose.** {purpose}","",
   f"[{MODELS[model]} {parameter} {sweep} (PDF)](figures/{filename(sweep,parameter)}.pdf)","",
   f"[Vector PDF](figures/{filename(sweep,parameter)}.pdf) | [Samples](data/{model}.csv.gz)","",text,""]
  if ko_rows is not None:
   reference={(r['coordinate'],r['parameter_value']):r for r in group_rows(ko_rows,sweep,parameter)}
   differences=[100*(r['eta_tt']-reference[(r['coordinate'],r['parameter_value'])]['eta_tt']) for r in selected if r['status']=='ok' and reference[(r['coordinate'],r['parameter_value'])]['status']=='ok']
   content += [f"**Model comparison.** Over the common finite samples, Benner minus KO total-total efficiency ranges from {min(differences):.2f} to {max(differences):.2f} percentage points. This measures model spread, not agreement with measurements.",""]
 directory=model_directory(model);directory.mkdir(exist_ok=True);(directory/'readme.md').write_text('\n'.join(content),encoding='utf-8')
 return results

METHOD=r"""# Single-stage air turbine: current loss-model trends

[Audit index](../readme.md) | [Shock and Reynolds formulation](../../kacker_okapuu_shock/readme.md)

__COMPARISON__

Air expands through one axial stage from **50 bar and 600 K total inlet conditions**. Kacker-Okapuu, Benner and the isentropic zero-loss model each have a separate full study with 21 figures on the same reference designs and baseline parameters. Both efficiencies use solid lines and separate blue/orange color scales.

__REPORTS__

## Baseline and parameter ranges

__BASELINE__

Seven parameters are varied one at a time in all three models. Each sweeps blade velocity ratio from 0 to 1.2 at $R=0$ and $R=0.5$, then reaction from -0.1 to 1.1 at $U/c_0=0.6$. Both rows share true-chord pitch ratio, aspect ratio and dimensionless trailing thickness. The specified stator angle baseline is 70 degrees; mass-flow levels use logarithmic color normalization.

Axial inlet flow, constant mean radius and equal meridional velocities define the reference triangles. Metal inlet angles follow inlet flow, giving zero incidence. Maximum thickness/chord is 0.30; rotor tip clearance/height is 0.01 and stator clearance is zero. Benner uses inlet displacement thickness/height 0.011. These are resized design families with continuous blade counts.

## Expansion and sizing

CoolProp HEOS Air supplies density, viscosity, sound speed and thermodynamic states. The isentropic expansion Mach number defines the exit pressure through

$$
c_0=\sqrt{2(h_{01}-h_{4s})},\qquad M_{\mathrm{is}}=c_0/a_{4s},\qquad s_{4s}=s_{01}.
$$

It describes the overall expansion; the saved row-frame Mach numbers are separate inputs to the loss functions. Specific speed sets angular velocity, mean radius and station heights:

$$
\Omega_s=\frac{\omega\sqrt{\dot m/\rho_{4s}}}{(h_{01}-h_{4s})^{3/4}},\qquad
r_m=\frac{U}{\omega},\qquad H_i=\frac{\dot m}{2\pi r_m\rho_i v_{m,i}}.
$$

Signed hub radii remain in geometry. Their loss-only hub/tip coordinate is identity at and above 0.5 and approaches **0.3** below that join. The same positive coordinate enters the hub-Mach fit and shock span weighting. Shock onset blends over hub Mach 0.4-0.5; the exit-normalized coefficient is exact through 0.25 and approaches **0.7** above it. The Reynolds coordinate equals the supplied value on $10^4\le\mathrm{Re}\le10^7$, tends to 1000 for large negative trial values, and to $5\times10^7$ at large positive values. Consequently its high-Re correction tends to $50^{-0.2}=0.457305052$.

The isentropic option sets every row-loss component to zero, so $\eta_{tt}=1$ and $\eta_{ts}=1-v_4^2/[2(h_{01}-h_{4s})]$. The latter can be below one because of exit kinetic energy. Geometry-only parameter curves overlap in this model; the expansion-Mach curves also overlap for the prescribed dimensionless triangles. No shock, Reynolds or mean-secant correction is applied to isentropic losses.

At zero blade speed, finite-specific-speed sizing is singular. At $R\ge1$ the chosen reference triangles lose real throughflow. Such failures remain gaps with explicit statuses. Negative hub radii and negative loss-corrected efficiencies are retained, without markers at their boundaries. All efficiency axes show 0-100%.

## Physical turning and mean-secant continuation

KO secondary and the clearance correction shared by both models use

$$
Z=4\left[\frac{\sin(\beta_{\mathrm{in}}-\beta_{\mathrm{out}})}{\cos\beta_{\mathrm{in}}}\right]^2\mathcal S(\sec\beta_m),
\qquad \tan\beta_m=\frac{\tan\beta_{\mathrm{in}}+\tan\beta_{\mathrm{out}}}{2}.
$$

The first factor equals $4[(w_{\theta,\mathrm{in}}-w_{\theta,\mathrm{out}})/w_{\mathrm{out}}]^2$ for these equal-meridional-velocity designs. It is calculated from physical angles, without independently saturating either tangent. The mean factor is unchanged through $\sec80^\circ$; above that join, with $\tau=(\sec\beta_m-\sec80^\circ)/(\sec87.5^\circ-\sec80^\circ)$, it is $\mathcal S=\sec87.5^\circ-(\sec87.5^\circ-\sec80^\circ)/(1+\tau+\tau^2)$. The join is C2 and the asymptote is a numerical choice that keeps the high-reaction limit finite without suppressing actual turning. There is no separate loading ceiling. KO secondary uses the physical projection $\cos\beta_{\mathrm{out}}/\cos\theta_{\mathrm{in}}$.

The angle-only loading domain is strictly $(-90^\circ,90^\circ)$. Exactly grazing and reversed-flow angles return NaN; no periodic extrapolation is assigned. This does not filter signed hub geometry. Profile reference angles retain their separate continuation. Benner retains its own secondary and penetration correlations; its shared clearance component uses the physical-turning formulation above. See the [secondary](../individual_functions/report/02_ko_secondary.md) and [clearance](../individual_functions/report/04_ko_clearance.md) notes for the equations and domain.

## Pressure-loss conversion and signed efficiency

The [core](../../../turbodash/core_turbine.py) supplies reference triangles and geometry; row losses come from the [current implementation](../../../turbodash/loss_models/). Each pressure-loss coefficient gives

$$
p_{0,\mathrm{rel,is}}=p_{0,\mathrm{rel,out}}+Y(p_{0,\mathrm{rel,out}}-p_{\mathrm{out}}),\qquad
s_{\mathrm{ref}}=s(h_{0,\mathrm{rel}},p_{0,\mathrm{rel,is}}),\qquad
\Delta h_{\mathrm{loss}}=h_{\mathrm{out}}-h(p_{\mathrm{out}},s_{\mathrm{ref}}).
$$

The signed efficiency definition is

$$
\Delta h_{0,\mathrm{ideal}}=h_{01}-h_{4s}-v_4^2/2,\qquad
\eta_{tt}=1-\frac{\sum\Delta h_{\mathrm{loss}}}{\Delta h_{0,\mathrm{ideal}}},\qquad
\eta_{ts}=\frac{\Delta h_{0,\mathrm{ideal}}-\sum\Delta h_{\mathrm{loss}}}{h_{01}-h_{4s}}.
$$

It uses isentropic reference triangles with an additive loss correction. Negative values remain in the data. The calculation does not update the Euler triangle with loss or solve passage choking, so transonic and extreme extrapolation trends require physical interpretation.

## Reproduction and evidence

[study.py](study.py) computes the designs and [report.py](report.py) writes figures and assessments.

```powershell
python -B loss_model_analysis/loss_model_audit/overall_stage/study.py
python -B loss_model_analysis/loss_model_audit/overall_stage/study.py --render-only
python -B loss_model_analysis/loss_model_audit/overall_stage/study.py --check
```

__COUNTS__

Checks cover the Mach boundary, Euler/reaction identities, geometry ratios, size similarity, EOS pressure/enthalpy round trips and the signed efficiency identity. Source files are hashed before and after calculation. [Provenance](data/provenance.json) | [Validation](data/validation.json) | [Assessments](data/assessments.json) | [Render hashes](data/render_manifest.json)
"""

def check_links(paths):
 count=0
 for path in paths:
  source=path.read_text(encoding='utf-8')
  targets=re.findall(r"!?\[[^\]]*\]\(([^)]+)\)",source)
  class HTMLLinks(HTMLParser):
   def handle_starttag(self,tag,attrs):
    values=dict(attrs)
    target=values.get('href' if tag=='a' else 'src') if tag in ('a','img') else None
    if target:targets.append(target)
  HTMLLinks().feed(source)
  for target in targets:
   if target.startswith(('http:','https:','#')):continue
   assert (path.parent/target.split('#')[0]).resolve().exists(),(path,target)
   count+=1
 return count

COMPARISON_MODELS=("isentropic","kacker_okapuu","benner")


def comparison_report(provenance,model_rows=None):
 """Write the matching sections in three columns using existing figures.

 Check matching configuration grids and section headings before comparing
 saved results. Source PDF files are linked, never regenerated.
 """
 if not set(COMPARISON_MODELS).issubset(provenance['models']):
  raise ValueError("The overall report requires saved isentropic, KO and Benner studies")
 rows=model_rows or {model:read_samples(model) for model in COMPARISON_MODELS}
 case_key=lambda row:(row['sweep'],row['parameter'],row['coordinate'],row['parameter_value'])
 signatures={model:{case_key(row):tuple(row[key] for key in CONFIG_KEYS)
                    for row in rows[model]} for model in COMPARISON_MODELS}
 assert all(signatures[model]==signatures['isentropic'] for model in COMPARISON_MODELS),"Model input grids differ"
 headings=[f"{PARAMETERS[parameter]['name']}: {title(sweep)}"
           for parameter in PARAMETERS for sweep in SWEEPS]
 for model in COMPARISON_MODELS:
  document=model_directory(model)/'readme.md'
  assert re.findall(r'^## (.+)$',document.read_text(encoding='utf-8'),re.M)==headings,document
 names={'isentropic':'Isentropic (zero losses)','kacker_okapuu':'Kacker–Okapuu','benner':'Benner'}
 content=['# Overall stage: side-by-side model sensitivities','',
  '[Common method and baseline](readme.md) | '+ ' | '.join(
   f'[{names[model]}]({model_directory(model).name}/readme.md)' for model in COMPARISON_MODELS),'',
  'Each section uses the same saved designs and parameter levels, with **isentropic, Kacker–Okapuu and Benner** in fixed left-to-right columns. '
  'Air enters at 50 bar and 600 K. All plots retain the same axes, 0–100% efficiency limits, blue total-total and orange total-static colormaps, and solid lines. '
  'Negative hub ratios remain in the data without boundary markers.','',
  'Open the vector PDF in each column to inspect the plot. The three models stay in one table row; a narrow preview may need horizontal scrolling. '
  'Isentropic total-total efficiency is exactly 100%; its total-static efficiency includes unrecovered exit kinetic energy. Coincident ideal curves are expected for parameters that only affect sizing or loss corrections.','']
 figure_hashes={};sections=[]
 for parameter in PARAMETERS:
  for sweep in SWEEPS:
   heading=f"{PARAMETERS[parameter]['name']}: {title(sweep)}"
   content += [f'## {heading}','',f'**Purpose.** {PURPOSES[parameter]}','',
    '<table width="100%" style="width:100%;table-layout:fixed">','<thead><tr>']
   content += [f'<th width="33.333%" align="center">{escape(names[model])}</th>' for model in COMPARISON_MODELS]
   content += ['</tr></thead>','<tbody><tr>']
   section_files=[]
   for model in COMPARISON_MODELS:
    stem=f'{model_directory(model).name}/figures/{filename(sweep,parameter)}'
    pdf=stem+'.pdf'
    path=BASE/pdf
    assert path.is_file(),path
    figure_hashes[pdf]=sha256(path)
    alt=escape(f'{names[model]}: {PARAMETERS[parameter]["name"]}, {sweep}',quote=True)
    content += ['<td width="33.333%" align="center" valign="top">',
     f'<a href="{pdf}" title="{alt}">Open vector PDF</a>','</td>']
    section_files.append(dict(model=model,pdf=pdf))
   content += ['</tr></tbody>','</table>','']
   groups={model:group_rows(rows[model],sweep,parameter) for model in COMPARISON_MODELS}
   common_keys=set.intersection(*[{case_key(row) for row in groups[model] if row['status']=='ok'} for model in COMPARISON_MODELS])
   anchor=.5 if sweep=='reaction' else .6
   coordinates=Counter(key[2] for key in common_keys)
   if coordinates:
    coordinate=min(coordinates,key=lambda value:(-coordinates[value],abs(value-anchor)))
    by_key={model:{case_key(row):row for row in groups[model]} for model in ('kacker_okapuu','benner')}
    common=[key for key in common_keys if key[2]==coordinate]
    differences={name:[100*(by_key['benner'][key][name]-by_key['kacker_okapuu'][key][name]) for key in common]
                 for name in ('eta_tt','eta_ts')}
    axis='$R$' if sweep=='reaction' else '$U/c_0$'
    comment=(f"**Comparison.** At {axis}={coordinate:g}, Benner minus KO spans "
             f"{min(differences['eta_tt']):.2f} to {max(differences['eta_tt']):.2f} percentage points in total-total efficiency and "
             f"{min(differences['eta_ts']):.2f} to {max(differences['eta_ts']):.2f} points in total-static efficiency across the common finite parameter levels.")
   else:comment='**Comparison.** No common finite samples exist in this family; retain the gaps shown in the individual reports.'
   content += [comment,'']
   sections.append(dict(heading=heading,sweep=sweep,parameter=parameter,columns=section_files))
 content += ['---','', '**Reproduce from saved figures.** No cases or plots are rerun:','',
  '```powershell','python -B loss_model_analysis/loss_model_audit/overall_stage/report.py --comparison-only','```','',
  '[Report and source-figure hashes](data/comparison_manifest.json)','']
 document=BASE/'comparison.md'
 document.write_text('\n'.join(content),encoding='utf-8')
 record=dict(passed=True,sections=len(sections),columns=3,model_order=list(COMPARISON_MODELS),
  matching_section_headings=True,matching_input_grids=True,figures_reused=len(figure_hashes),
  model_calculations=0,plots_regenerated=0,figure_sha256=figure_hashes,
  data_sha256=provenance['data_sha256'],report_sha256=sha256(document),
  report_script_sha256=sha256(__file__),section_inventory=sections)
 write_json(DATA/'comparison_manifest.json',record)
 record['checked_links']=check_links([document])
 assert all(sha256(BASE/path)==digest for path,digest in figure_hashes.items()),'Source figures changed'
 write_json(DATA/'comparison_manifest.json',record)
 return record


def link_comparison():
 """Link the three-column report from an existing common-method index."""
 path=BASE/'readme.md';text=path.read_text(encoding='utf-8')
 link='[Side-by-side comparison: isentropic, Kacker–Okapuu and Benner](comparison.md)'
 if '(comparison.md)' not in text:
  lines=text.splitlines();lines[2:2]=[link,''];path.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def generate(provenance):
 style();model_rows={m:read_samples(m) for m in provenance['models']};outputs={};assessments={}
 for model,rows in model_rows.items():
  pairs=[(s,p) for p in PARAMETERS for s in SWEEPS]
  for sweep,parameter in pairs:
   outputs.update(plot_group(model,sweep,parameter,group_rows(rows,sweep,parameter)))
  assessments[model]=model_report(model,rows)
 reports=['| Model | Report | Samples |','|---|---|---|']
 for model,rows in model_rows.items():reports.append(f"| {MODELS[model]} | [Report]({model_directory(model).name}/readme.md) | {len(rows):,} |")
 baseline=['| Parameter | Range | Baseline |','|---|---:|---:|']
 for setting in PARAMETERS.values():baseline.append(f"| {setting['name']} {setting['symbol']} | {setting['values'][0]:g} to {setting['values'][-1]:g} {setting['unit']} | {setting['baseline']:g} {setting['unit']} |")
 text=METHOD.replace('__REPORTS__','\n'.join(reports)).replace('__BASELINE__','\n'.join(baseline))
 text=text.replace('__COMPARISON__','[Side-by-side model comparison](comparison.md)' if set(COMPARISON_MODELS).issubset(model_rows) else '')
 text=text.replace('__COUNTS__',f"The saved run contains {sum(len(r) for r in model_rows.values()):,} samples in {len(outputs)} PDF figures. Each model retains all seven sensitivity families and all three sweeps.")
 (BASE/'readme.md').write_text(text,encoding='utf-8');write_json(DATA/'assessments.json',assessments)
 if set(COMPARISON_MODELS).issubset(model_rows):comparison_report(provenance,model_rows)
 write_render_manifest(provenance,outputs)
 print(f"Wrote {len(outputs)} current-model PDF figures",flush=True)


def write_render_manifest(provenance,outputs=None):
 """Validate completed figures/reports and publish their current hashes."""
 if outputs is None:
  outputs={}
  for model in provenance['models']:
   for parameter in PARAMETERS:
    for sweep in SWEEPS:
     for extension in ("pdf",):
      path=model_directory(model)/'figures'/f'{filename(sweep,parameter)}.{extension}'
      outputs[path.relative_to(BASE).as_posix()]=sha256(path)
 documents=[BASE/'readme.md',*(model_directory(m)/'readme.md' for m in provenance['models'])]
 if (BASE/'comparison.md').exists():documents.append(BASE/'comparison.md')
 manifest=dict(figures=len(outputs),figure_sha256=outputs,
  report_sha256={p.relative_to(BASE).as_posix():sha256(p) for p in documents},report_script_sha256=sha256(__file__),
  data_sha256=provenance['data_sha256'],source_matches=source_hashes()==provenance['source_sha256'],
  model_calculations=0,hub_ratio_markers=False)
 # Bootstrap the manifest before checking the report's link to that file.
 write_json(DATA/'render_manifest.json',manifest)
 manifest['checked_links']=check_links(documents)
 write_json(DATA/'render_manifest.json',manifest)
 assessments=json.loads((DATA/'assessments.json').read_text(encoding='utf-8'))
 for model in provenance['models']:
  model_data=model_directory(model)/'data'
  prefix=model_directory(model).name+'/'
  model_hashes={str(key).removeprefix(prefix):digest for key,digest in outputs.items() if key.startswith(prefix)}
  write_json(model_data/'render_manifest.json',dict(figures=len(model_hashes),
   figure_sha256=model_hashes,report_sha256=sha256(model_directory(model)/'readme.md'),
   report_script_sha256=sha256(__file__),hub_ratio_markers=False,
   physical_turning=model!="isentropic",physical_secondary_projection=model=='kacker_okapuu',
   mean_secant_onset_degrees=None if model=="isentropic" else 80.,
   mean_secant_limit_degrees=None if model=="isentropic" else 87.5,
   data_sha256={f'{model}.csv.gz':sha256(model_data/f'{model}.csv.gz')}))
  write_json(model_data/'provenance.json',dict(provenance,model=model,
   data_sha256={f'{model}.csv.gz':sha256(model_data/f'{model}.csv.gz')},
   common_provenance='../../data/provenance.json'))
  write_json(model_data/'summary.json',json.loads((DATA/'summary.json').read_text(encoding='utf-8'))[model])
  write_json(model_data/'validation.json',json.loads((DATA/'validation.json').read_text(encoding='utf-8')))
  write_json(model_data/'assessments.json',assessments[model])


if __name__=='__main__':
 import argparse
 from study import verify_saved
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--comparison-only',action='store_true',required=True,
                     help='Build the three-column report from existing PDF figures and samples')
 parser.parse_args()
 provenance=verify_saved()
 result=comparison_report(provenance)
 link_comparison()
 write_render_manifest(provenance)
 print(f"Wrote comparison.md: {result['sections']} sections, three columns; no cases or plots rerun.")
