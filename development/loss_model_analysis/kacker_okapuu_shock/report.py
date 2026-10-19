"""Figures and report for the isolated signed-hub shock-continuation study."""
import json
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MultipleLocator

from study import (BASE, DATA, SPECIFIC_SPEEDS, VARIANTS, RECOMMENDED, BASELINE,
                   JOIN, continued_ratio, candidate_shock, ko, sha256, write_json,
                   read_samples, sources, reference)
from development.loss_model_analysis.pdf_sources import chart_image

FIGURES=BASE/"figures"
INK="#172c43"
MUTED="#596d7f"


def style():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10.,"mathtext.fontset":"dejavusans",
        "axes.labelcolor":INK,"text.color":INK,"xtick.color":MUTED,"ytick.color":MUTED,
        "axes.edgecolor":"#c4cfd7","axes.spines.top":False,"axes.spines.right":False,
        "axes.facecolor":"white","figure.facecolor":"white","grid.color":"#dce3e9",
        "grid.alpha":.65,"grid.linewidth":.7,"svg.fonttype":"path","savefig.facecolor":"white"})


def palettes():
    return [LinearSegmentedColormap.from_list(name+"_readable",plt.get_cmap(name)(np.linspace(.4,.96,256)))
            for name in ("Blues","Oranges")]


def select(rows,variant=RECOMMENDED,specific_speed=1.05):
    return sorted([r for r in rows if r["variant"]==variant and r["specific_speed"]==specific_speed],key=lambda r:r["speed"])


def array(curve,key):
    return np.array([r[key] for r in curve])


def point(rows,speed,variant=RECOMMENDED,specific_speed=1.05):
    curve=select(rows,variant,specific_speed)
    row=min(curve,key=lambda r:abs(r["speed"]-speed))
    assert abs(row["speed"]-speed)<1e-9
    return row


def decorate(ax,xlabel=r"Blade velocity ratio $U/c_0$",xlim=(0,.9)):
    ax.set_xlabel(xlabel,labelpad=8)
    ax.set_xlim(*xlim)
    ax.grid(True)
    ax.tick_params(length=0,pad=6)


def save(fig,name):
    result={}
    for extension in ("svg",):
        path=FIGURES/f"{name}.{extension}"
        fig.savefig(path,dpi=190,metadata={"Creator":"Turbodash shock continuation study"} if extension=="svg" else None)
        result[path.name]=sha256(path)
    plt.close(fig)
    return result


def heading(fig,title,subtitle):
    fig.suptitle(title,x=.075,y=.975,ha="left",size=20,weight="bold")
    fig.text(.075,.914,subtitle,size=10.5,color=MUTED)


def family_colorbars(fig,blue,orange=None):
    norm=Normalize(SPECIFIC_SPEEDS[0],SPECIFIC_SPEEDS[-1])
    entries=[(.085,.87,blue,r"Specific speed $\Omega_s$")]
    if orange is not None:
        entries=[(.085,.405,blue,r"Total–total $\eta_{tt}$"),(.55,.405,orange,r"Total–static $\eta_{ts}$")]
    for left,width,cmap,title in entries:
        cax=fig.add_axes([left,.13,width,.018])
        cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation="horizontal")
        cb.set_ticks(SPECIFIC_SPEEDS,labels=[f"{v:g}" for v in SPECIFIC_SPEEDS])
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=9,length=3,color=MUTED)
        fig.text(left,.179,title,size=11,weight="bold",color=cmap(.8))
    if orange is not None:
        fig.text(.52,.052,r"Curve parameter: specific speed $\Omega_s$  ·  bold: 1.05  ·  both efficiencies use solid lines",
                 ha="center",size=9,color=MUTED)


def efficiency_figure(rows):
    blue,orange=palettes(); norm=Normalize(.1,2)
    fig=plt.figure(figsize=(11.6,7.3)); ax=fig.add_axes([.085,.33,.87,.49])
    heading(fig,"Efficiency through the zero-hub boundary",r"KO · $R=0$ · $\alpha_2=70^\circ$ · shared coordinate (0.30 limit) · original efficiency definition")
    for ns in SPECIFIC_SPEEDS:
        curve=select(rows,specific_speed=ns); speed=array(curve,"speed")
        for key,cmap in (("eta_tt_linear",blue),("eta_ts_linear",orange)):
            values=100*array(curve,key)
            ax.plot(speed,values,color=cmap(norm(ns)),lw=2.7 if ns==1.05 else 1.7)
    decorate(ax); ax.set_ylabel("Efficiency [%]",labelpad=9); ax.set_ylim(0,100)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    family_colorbars(fig,blue,orange)
    fig.text(.085,.228,"Signed hub radii and efficiencies are retained; efficiency axes show 0-100%.",size=9.5,color=MUTED)
    return save(fig,"01_efficiency")


def geometry_figure(rows):
    blue,_=palettes();norm=Normalize(.1,2)
    fig,axes=plt.subplots(1,2,figsize=(12,5.8))
    fig.subplots_adjust(left=.08,right=.97,bottom=.33,top=.79,wspace=.25)
    heading(fig,"Raw geometry and the loss coordinate","The signed hub radius stays in the geometry. Only the loss correlation receives a continued coordinate.")
    for ns in SPECIFIC_SPEEDS:
        curve=select(rows,specific_speed=ns)
        axes[0].plot(array(curve,"speed"),array(curve,"rotor_raw_hub_tip"),color=blue(norm(ns)),lw=2.6 if ns==1.05 else 1.7)
    axes[0].axhspan(-1,0,color="#e9c98d",alpha=.16)
    axes[0].axhline(0,color=MUTED,lw=.8)
    decorate(axes[0]);axes[0].set_ylabel(r"Raw rotor inlet $r_h/r_t$");axes[0].set_ylim(-1.05,1.05)
    ratios=np.linspace(-1,1,1001)
    axes[1].plot(ratios,ratios,color="#a0a9b1",lw=1.3,label="Raw ratio")
    axes[1].plot(ratios,continued_ratio(ratios),color="#176d8f",lw=2,label="Positive loss coordinate / limit 0.30")
    axes[1].axvline(.5,color=MUTED,lw=.8,ls=":")
    decorate(axes[1],r"Raw $r_h/r_t$",(-1,1));axes[1].set_ylabel(r"Loss coordinate $(r_h/r_t)_{\mathrm{eff}}$")
    axes[1].set_ylim(-1.05,1.05);axes[1].legend(frameon=False,fontsize=9,loc="lower right")
    family_colorbars(fig,blue)
    return save(fig,"02_geometry")


def shock_law_figure():
    components=json.loads((DATA/"components.json").read_text(encoding="utf-8"))
    ratios=np.array(components["ratios"])
    fig,axes=plt.subplots(1,2,figsize=(12,5.8));fig.subplots_adjust(left=.08,right=.97,bottom=.21,top=.79,wspace=.25)
    heading(fig,"Shock loss with a positive hub/tip coordinate",r"Fixed inlet Mach 0.8 / inlet-to-exit pressure-head ratio = 1")
    for family,color in (("stator","#176d8f"),("rotor","#d9671c")):
        for ax,key in zip(axes,("Y_shock","derivative")):
            ax.plot(ratios,components[family][key],color=color,lw=2,label=family.capitalize())
    for ax in axes:
        decorate(ax,r"Raw inlet $r_h/r_t$",(-1,1));ax.axvline(.5,color=MUTED,lw=.8,ls=":")
        ax.legend(frameon=False)
    axes[0].set(ylim=(0,.72),ylabel=r"Shock coefficient $Y_{p,\mathrm{shock}}$")
    axes[0].axhline(.7,color=MUTED,lw=.8,ls=":")
    axes[1].set_ylabel(r"$dY_{p,\mathrm{shock}}/d(r_h/r_t)$")
    fig.text(.08,.065,"The same continued coordinate enters the hub-Mach fit and span weighting; signed geometry remains available.",size=9,color=MUTED)
    return save(fig,"03_shock_components")


def components_figure(rows):
    blue,_=palettes();norm=Normalize(.1,2)
    fig,axes=plt.subplots(2,2,figsize=(12,8.8));fig.subplots_adjust(left=.08,right=.97,bottom=.29,top=.85,hspace=.4,wspace=.25)
    heading(fig,"Shock and profile loss at low blade speed","Shared coordinate (0.30 limit) · the shock term includes its production upper continuation")
    for ns in SPECIFIC_SPEEDS:
        curve=select(rows,specific_speed=ns);speed=array(curve,"speed")
        for ax,key in zip(axes.flat,("stator_shock","rotor_shock","stator_profile","rotor_profile")):
            ax.plot(speed,array(curve,key),color=blue(norm(ns)),lw=2.6 if ns==1.05 else 1.6)
    for ax,title in zip(axes.flat,("Stator: shock before Reynolds scaling","Rotor: shock before Reynolds scaling",
                                   "Stator: complete profile loss","Rotor: complete profile loss")):
        decorate(ax);ax.set_title(title,loc="left",size=11,weight="bold");ax.set_ylim(bottom=0)
    for ax in axes[0]:ax.set_ylabel(r"$Y_{p,\mathrm{shock}}$")
    for ax in axes[1]:ax.set_ylabel(r"$Y_p$")
    family_colorbars(fig,blue)
    return save(fig,"04_profile_and_shock")


def budget_figure(rows):
    curve=select(rows); speed=array(curve,"speed")
    fig,axes=plt.subplots(1,2,figsize=(12,5.8))
    fig.subplots_adjust(left=.085,right=.97,bottom=.21,top=.79,wspace=.28)
    heading(fig,"Keep the original signed efficiency",r"$\Omega_s=1.05$ · losses and ideal work use the same reference triangles")
    axes[0].plot(speed,array(curve,"ideal_work")/1000,color="#176d8f",lw=2,label="Ideal shaft work")
    axes[0].plot(speed,array(curve,"total_enthalpy_loss")/1000,color="#b17b24",lw=2,label="Combined enthalpy-loss estimate")
    axes[0].set_ylabel("Specific energy [kJ/kg]");axes[0].set_ylim(bottom=0)
    axes[1].plot(speed,100*array(curve,"eta_tt_linear"),color="#176d8f",lw=2,label=r"$\eta_{tt}$")
    axes[1].plot(speed,100*array(curve,"eta_ts_linear"),color="#d9671c",lw=2,label=r"$\eta_{ts}$")
    axes[1].set_ylabel("Efficiency [%]");axes[1].set_ylim(0,100)
    for ax in axes:
        decorate(ax);ax.legend(frameon=False,fontsize=9)
    fig.text(.085,.065,"Efficiency becomes negative when the loss estimate exceeds ideal shaft work. Those samples remain in the data.",size=9,color=MUTED)
    return save(fig,"05_efficiency_budget")


def limit_figure(rows):
    curve=select(rows);speed=array(curve,"speed")
    fig,axes=plt.subplots(2,2,figsize=(12,8.5));fig.subplots_adjust(left=.085,right=.97,bottom=.09,top=.85,hspace=.38,wspace=.28)
    heading(fig,"What remains as blade speed approaches zero?",r"$\Omega_s=1.05$ · all samples retained down to $U/c_0=10^{-5}$ · shared coordinate (0.30 limit)")
    for key,label,color in (("rotor_shock","Shock before profile scaling","#7659a5"),
                            ("rotor_profile_shock","Shock inside profile loss","#b17b24"),
                            ("rotor_profile_base","Profile without shock","#328ca0"),
                            ("rotor_profile","Complete profile loss","#172c43")):
        axes[0,0].plot(speed,array(curve,key),label=label,color=color,lw=1.8)
    axes[0,0].set_ylabel("Rotor pressure-loss coefficient");axes[0,0].legend(frameon=False,fontsize=8.5)
    axes[0,1].plot(speed,array(curve,"rotor_f_Re"),color="#176d8f",lw=2)
    axes[0,1].set_ylabel(r"Rotor Reynolds correction $f_{\mathrm{Re}}$")
    axes[1,0].plot(speed,array(curve,"ideal_work")/1000,label="Ideal shaft work",color="#176d8f",lw=2)
    axes[1,0].plot(speed,array(curve,"total_enthalpy_loss")/1000,label="Combined enthalpy-loss estimate",color="#b17b24",lw=2)
    axes[1,0].set_ylabel("Specific energy [kJ/kg]");axes[1,0].legend(frameon=False,fontsize=9)
    axes[1,1].plot(speed,100*array(curve,"eta_tt_linear"),color="#176d8f",label=r"$\eta_{tt}$",lw=2)
    axes[1,1].plot(speed,100*array(curve,"eta_ts_linear"),color="#d9671c",label=r"$\eta_{ts}$",lw=2)
    axes[1,1].set_ylabel("Efficiency [%]");axes[1,1].legend(frameon=False)
    for ax in axes.flat:
        decorate(ax,xlim=(1e-5,.9));ax.set_xscale("log");ax.set_ylim(bottom=0)
    axes[1,1].set_ylim(0,100)
    return save(fig,"06_low_speed_limit")


def reynolds_figure(rows):
    curve=select(rows);speed=array(curve,"speed")
    fig,axes=plt.subplots(2,2,figsize=(12,8.5));fig.subplots_adjust(left=.085,right=.97,bottom=.12,top=.84,hspace=.38,wspace=.27)
    heading(fig,"Finite Reynolds correction at very low speed",r"Specific speed 1.05 / effective Reynolds limit = fifty million")
    for column,row in enumerate(("stator","rotor")):
        color="#176d8f" if row=="stator" else "#d9671c"
        axes[0,column].plot(speed,array(curve,f"{row}_Re"),color="#929da7",lw=1.5,label="Supplied Reynolds number")
        axes[0,column].plot(speed,array(curve,f"{row}_Re_effective"),color=color,lw=2.3,label="Correlation coordinate")
        axes[0,column].set_yscale("log");axes[0,column].set_ylabel("Reynolds number")
        axes[0,column].set_title(row.capitalize(),loc="left",size=12,weight="bold");axes[0,column].legend(frameon=False,fontsize=8.5)
        axes[1,column].plot(speed,array(curve,f"{row}_f_Re"),color=color,lw=2.3)
        axes[1,column].axhline(50**(-.2),color=MUTED,lw=.8,ls=":")
        axes[1,column].set(ylabel=r"Reynolds correction $f_{\mathrm{Re}}$",ylim=(0,1.1))
        for ax in axes[:,column]:decorate(ax,xlim=(1e-5,.9));ax.set_xscale("log")
    fig.text(.085,.035,r"Dotted line: $f_{\mathrm{Re}}\to50^{-0.2}=0.457305$. Both reference-profile and shock contributions retain this multiplier.",size=9,color=MUTED)
    return save(fig,"07_reynolds_correction")


def calibration_figure():
    import csv
    with (DATA/"digitized_points.csv").open(encoding="utf-8",newline="") as stream:points=list(csv.DictReader(stream))
    components=json.loads((DATA/"components.json").read_text(encoding="utf-8"));ratios=np.array(components["ratios"])
    fig,axes=plt.subplots(1,2,figsize=(12,6));fig.subplots_adjust(left=.08,right=.97,bottom=.16,top=.8,wspace=.25)
    heading(fig,"Hub-Mach fit against the source chart","KO Figure 6 / 32 digitized points / calibration interval starts at hub/tip 0.5")
    axes[0].imshow(chart_image("ko_hub_mach"));axes[0].axis("off")
    for family,color in (("stator","#176d8f"),("rotor","#d9671c")):
        selected=[p for p in points if p["family"]==family]
        good=ratios>=.5
        axes[1].plot(ratios[good],np.array(components[family]["f_hub"])[good],color=color,lw=2,label=family.capitalize())
        axes[1].scatter([float(p["hub_tip_ratio"]) for p in selected],[float(p["hub_mach_factor"]) for p in selected],color=color,s=13,zorder=4)
    decorate(axes[1],r"Inlet $r_h/r_t$",(.5,1));axes[1].set_ylabel(r"Hub/mean Mach ratio $f_{\mathrm{hub}}$");axes[1].legend(frameon=False)
    fig.text(.08,.045,"The continued low-ratio region is a modeling choice; chart agreement is assessed only on its digitized interval.",size=9,color=MUTED)
    return save(fig,"08_hub_mach_fit")


def metrics(rows):
    return dict(validation=json.loads((DATA/"validation.json").read_text(encoding="utf-8")))


def write_report(rows,provenance,review):
    tiny=point(rows,1e-5)
    table=[r"| Specific speed $\Omega_s$ | Raw rotor $r_h/r_t$ | Rotor shock | $\eta_{tt}$ [%] | $\eta_{ts}$ [%] |","|---:|---:|---:|---:|---:|"]
    for ns in SPECIFIC_SPEEDS:
        r=point(rows,.05,specific_speed=ns)
        table.append(f"| {ns:g} | {r['rotor_raw_hub_tip']:.4f} | {r['rotor_shock']:.4f} | {100*r['eta_tt_linear']:.2f} | {100*r['eta_ts_linear']:.2f} |")
    text=METHOD.replace('__TABLE__','\n'.join(table))
    for key,value in dict(SAMPLES=f"{provenance['samples']:,}",NEGATIVE=f"{int(review['validation']['negative_hub_designs_retained']):,}",
        STRESS=f"{review['validation']['stress_cases']:,}",STATOR=f"{tiny['stator_profile']:.4f}",ROTOR=f"{tiny['rotor_profile']:.4f}").items():text=text.replace(f'__{key}__',value)
    (BASE/'readme.md').write_text(text,encoding='utf-8')


METHOD=r"""# Kacker-Okapuu inlet shock and Reynolds continuation

This study evaluates the [production shock and Reynolds helpers](../../../turbodash/loss_models/loss_model_kacker_okapuu.py). One positive hub/tip coordinate enters both the hub-Mach fit and shock span weighting. Its lower asymptote is **0.3**, joined to the physical chart coordinate at **0.5**. The shock coefficient approaches **0.7**. The Reynolds correction approaches **0.457305052**, the empirical high-Re power-law value at fifty million. Signed hub geometry and signed efficiencies are retained.

## 1. Hub-Mach fit and source evidence

**Purpose.** Check the row-specific fit against KO Figure 6 without treating extrapolated points as calibration data.

![Digitized hub-Mach chart and fitted production curves (SVG)](figures/08_hub_mach_fit.svg)

[Vector SVG](figures/08_hub_mach_fit.svg) | [Digitized readings](data/digitized_points.csv) | [Fit errors](data/fit_parameters.json)

The [KO paper](source/kacker_okapuu_1982.pdf#page=3), printed page 113, supplies Figure 6 and equations (4)-(6). The source fit is $f_{\mathrm{hub},j}=1+A_j[P_{0.01}(c_j-(r_h/r_t)_{\mathrm{eff}})]^2$, using a compact C2 positive part. Stator parameters are $A_j=4.348048474772493$, $c_j=0.8084516270970612$; rotor parameters are $A_j=5.946253981890890$, $c_j=0.9468539887756984$.

**Assessment.** The digitized interval begins at raw ratio 0.5. The plot checks the fitted correlation there; values below that interval use the explicit continuation described next.

## 2. Signed geometry and a positive loss coordinate

**Purpose.** Keep negative hub radii available for blade visualization while giving the empirical correlation a positive input.

$$
\left(\frac{r_h}{r_t}\right)_{\mathrm{eff}}=
\begin{cases}
\dfrac{r_h}{r_t},&r_h/r_t\ge0.5,\\[6pt]
0.3+\dfrac{0.2}{1+\dfrac{0.5-r_h/r_t}{0.2}+\left(\dfrac{0.5-r_h/r_t}{0.2}\right)^2},&r_h/r_t<0.5.
\end{cases}
$$

It matches value, slope and curvature at 0.5 and approaches 0.3 as the raw ratio tends to negative infinity. At raw zero it is 0.320513. At raw ratio -1 it is 0.303089. Raw geometry is retained.

![Raw geometry and continued loss coordinate (SVG)](figures/02_geometry.svg)

[Vector SVG](figures/02_geometry.svg)

**Assessment.** The loss input stays positive through zero and negative raw ratios. The map is applied once to the hub-Mach fit, and the identical coordinate supplies the span weight. Its limits are modeling choices, with chart behavior retained at and above 0.5.

## 3. Complete inlet shock coefficient

**Purpose.** Check the shock term and its derivative independently of changing stage triangles.

The hub Mach is $M_{1,\mathrm{hub}}=f_{\mathrm{hub}}M_1$. Shock is zero through hub Mach 0.4; a quintic smoothstep over 0.4-0.5 joins the power $(M_{1,\mathrm{hub}}-0.4)^{1.75}$. Above that blend,

$$
Y_{p,\mathrm{shock,raw}}=0.75\left(\frac{r_h}{r_t}\right)_{\mathrm{eff}}
(M_{1,\mathrm{hub}}-0.4)^{1.75}
\frac{p_{0,\mathrm{rel,in}}-p_{\mathrm{in}}}{p_{0,\mathrm{rel,out}}-p_{\mathrm{out}}}.
$$

The coefficient is retained through 0.25. Above that join,

$$
Y_{p,\mathrm{shock}}=0.7-\frac{0.45}{1+\dfrac{Y_{p,\mathrm{shock,raw}}-0.25}{0.45}
+\left(\dfrac{Y_{p,\mathrm{shock,raw}}-0.25}{0.45}\right)^2}.
$$

![Production shock coefficient and derivative (SVG)](figures/03_shock_components.svg)

[Vector SVG](figures/03_shock_components.svg) | [Component values](data/components.json)

**Assessment.** The coefficient is finite and nonnegative for valid pressure heads, including signed hub ratios. The same positive coordinate in both shock factors preserves an active shock as the raw hub ratio crosses zero. The 0.7 upper asymptote controls small-exit-head amplification before the outer KO/Reynolds scaling. Pressure heads remain the supplied thermodynamic values.

## 4. Efficiency versus blade velocity ratio

**Purpose.** Check one impulse-stage air-turbine family across five specific speeds, retaining negative hub radii and negative efficiencies.

Total inlet conditions are 50 bar and 600 K; $R=0$, stator exit angle $70^\circ$, $s/c=0.725$, $H/c=1.75$, $t_{\mathrm{TE}}/o=0.2$, mass flow 10 kg/s and overall isentropic expansion Mach 1.15. Specific speeds are 0.1, 0.5, 1.05, 1.5 and 2.0. The [stage study](../loss_model_audit/overall_stage/readme.md) defines the EOS and resizing method.

![Efficiencies across specific speed (SVG)](figures/01_efficiency.svg)

[Vector SVG](figures/01_efficiency.svg)

**Assessment.** Both efficiencies use solid lines, with blue and orange colormaps. Axes show 0-100%; signed values outside those limits remain saved. No hub-ratio markers are drawn. At blade velocity ratio 0.05:

__TABLE__

## 5. Complete profile and shock loss

**Purpose.** Check the combined profile assembly and identify the role of its Reynolds multiplier.

$$
Y_p=0.914f_{\mathrm{Re}}\left[\frac23K_pY_{p,\mathrm{reference}}F_t+Y_{p,\mathrm{shock}}\right].
$$

![Complete profile and shock loss (SVG)](figures/04_profile_and_shock.svg)

[Vector SVG](figures/04_profile_and_shock.svg)

**Assessment.** The unscaled shock and complete profile remain finite at very low blade speed. Both profile contributions receive the same nonzero high-Reynolds multiplier. Secondary, trailing-edge and clearance coefficients are evaluated through their production helpers.

## 6. Reynolds correction at extreme size

**Purpose.** Verify that the large-chord Reynolds numbers created by this sizing family retain finite loss coefficients.

On $10^4\le\mathrm{Re}\le10^7$, the correction uses the supplied Reynolds number. Its compact C2 blending bands are centered at $2\times10^5$ and $10^6$. Below $10^4$ the effective coordinate approaches 1000 for negative solver iterates. Above $10^7$,

$$
\mathrm{Re}_{\mathrm{eff}}=5\times10^7-
\frac{4\times10^7}{1+\dfrac{\mathrm{Re}-10^7}{4\times10^7}+\left(\dfrac{\mathrm{Re}-10^7}{4\times10^7}\right)^2},\qquad
f_{\mathrm{Re}}\longrightarrow50^{-0.2}=0.457305052.
$$

![Current Reynolds coordinate and correction (SVG)](figures/07_reynolds_correction.svg)

[Vector SVG](figures/07_reynolds_correction.svg)

**Assessment.** Value, slope and curvature match at the coordinate joins. The finite factor keeps both reference and shock losses present as blade size grows. At blade velocity ratio $10^{-5}$ and specific speed 1.05, complete stator and rotor profile coefficients are **__STATOR__** and **__ROTOR__**. The limits are extrapolation choices.

## 7. Signed efficiency budget and low-speed limit

**Purpose.** Explain the efficiency curves using the specified loss budget.

$$
\Delta h_{0,\mathrm{ideal}}=h_{01}-h_{4s}-v_4^2/2,\qquad
\eta_{tt}=1-\frac{\sum\Delta h_{\mathrm{loss}}}{\Delta h_{0,\mathrm{ideal}}},\qquad
\eta_{ts}=\frac{\Delta h_{0,\mathrm{ideal}}-\sum\Delta h_{\mathrm{loss}}}{h_{01}-h_{4s}}.
$$

![Signed efficiency and energy budget (SVG)](figures/05_efficiency_budget.svg)

[Vector SVG](figures/05_efficiency_budget.svg)

![Low-speed limit of loss, Reynolds correction, energy and efficiency (SVG)](figures/06_low_speed_limit.svg)

[Vector SVG](figures/06_low_speed_limit.svg)

**Assessment.** Ideal shaft work tends to zero while the loss estimate remains finite. The signed total-total efficiency therefore becomes negative. All values remain in the data; the efficiency axes show only 0-100%.

## Reproduction and checks

The active calculation and plotting scripts are [study.py](study.py) and [report.py](report.py). [digitize.py](digitize.py) records chart readings and [render_source.py](render_source.py) supplies paper crops. Calculations call the production helpers directly.

```powershell
python -B development/loss_model_analysis/kacker_okapuu_shock/study.py
python -B development/loss_model_analysis/kacker_okapuu_shock/study.py --render-only
python -B development/loss_model_analysis/kacker_okapuu_shock/study.py --check
```

The run contains **__SAMPLES__ samples**, including **__NEGATIVE__ negative-hub designs**. Checks cover **__STRESS__ shock stress cases**, 1,001 high-Reynolds cases through $10^{100}$, geometry ratios, full-profile assembly, C2 coordinate joins and the signed efficiency identity. [Validation](data/validation.json) | [Provenance](data/provenance.json) | [Render hashes](data/render_manifest.json)
"""


def check_links():
    text=(BASE/"readme.md").read_text(encoding="utf-8")
    checked=0
    for link in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)",text):
        if link.startswith(("http:","https:","#")):continue
        assert (BASE/link.split("#")[0]).resolve().exists(),link
        checked+=1
    return checked


def generate(provenance):
    style();FIGURES.mkdir(parents=True,exist_ok=True)
    rows=read_samples();outputs={}
    for plot in (efficiency_figure,geometry_figure,components_figure,budget_figure,limit_figure,reynolds_figure):
        outputs.update(plot(rows))
    outputs.update(shock_law_figure())
    outputs.update(calibration_figure())
    review=metrics(rows)
    write_json(DATA/"review.json",review)
    write_report(rows,provenance,review)
    manifest=dict(figures=len(outputs),figure_sha256=outputs,report_sha256=sha256(BASE/"readme.md"),
                  report_script_sha256=sha256(__file__),data_sha256=provenance["data_sha256"],
                  sources_match_worktree=sources()==provenance["source_sha256"])
    write_json(DATA/"render_manifest.json",manifest)
    manifest["checked_links"]=check_links()
    write_json(DATA/"render_manifest.json",manifest)
    print(f"Wrote {len(outputs)} SVG figures and the report; {manifest['checked_links']} links checked",flush=True)
