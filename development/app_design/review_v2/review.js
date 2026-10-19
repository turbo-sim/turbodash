/* Offline interaction study. No Dash connection, numerical solver or design-file IO. */
"use strict";
const $ = (selector, parent = document) => parent.querySelector(selector);
const $$ = (selector, parent = document) => [...parent.querySelectorAll(selector)];
const decisions = JSON.parse($("#decision-data").textContent);
const feedbackKey = "turbodash.design-review.v2.feedback";
const themeKey = "turbodash.design-review.v2.theme";
let saved = {};
try { saved = JSON.parse(localStorage.getItem(feedbackKey) || "{}"); } catch (_) {}
for (const d of decisions.decisions) {
  const prior = saved.decisions?.find(x => x.id === d.id);
  if (prior) { d.choice = d.options.includes(prior.choice) ? prior.choice : null; d.notes = String(prior.notes || ""); }
}
let generalFeedback = saved.generalFeedback || "";
let toastTimer;
function toast(message) {
  $("#toast").textContent = message;
  $("#toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $("#toast").hidden = true, 4000);
}
function choice(id) { return decisions.decisions.find(d => d.id === id)?.choice; }
function saveFeedback() {
  const payload = {version: 2, kind: "design feedback, not production settings", decisions: decisions.decisions,
    generalFeedback, updatedAt: new Date().toISOString()};
  try { localStorage.setItem(feedbackKey, JSON.stringify(payload)); $("#feedback-status").textContent = "Saved in this browser. Export the JSON to send your choices and notes back."; }
  catch (_) { $("#feedback-status").textContent = "Browser storage is unavailable. Export the JSON to keep your feedback."; }
  $("#pending-count").textContent = decisions.decisions.filter(d => !d.choice).length;
  return payload;
}
function updateBrand() {
  const style = choice("D02");
  const slug = style === "Axial cascade symbol" ? "cascade" : style === "TD monogram" ? "monogram" : "blade-ring";
  $("#brand-logo").src = `assets/${slug}-${document.documentElement.dataset.theme}.svg`;
  const long = choice("D04")?.includes("Meanline design");
  $$(".workspace-nav button").forEach(b => b.textContent = (long ? {meanline:"Meanline design",geometry:"Geometry generation",mesh:"Mesh generation"} : {meanline:"Meanline",geometry:"Geometry",mesh:"Mesh"})[b.dataset.workspace]);
  const dark = choice("D01");
  document.documentElement.dataset.darkPalette = dark?.includes("Midnight") ? "midnight" : dark?.includes("cyan") ? "cyan" : "graphite";
}
function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  $$(".theme-switch button").forEach(b => { const active = b.dataset.theme === theme; b.classList.toggle("active", active); b.setAttribute("aria-pressed", String(active)); });
  try { localStorage.setItem(themeKey, theme); } catch (_) {}
  updateBrand();
}
let initialTheme = "light";
try { initialTheme = localStorage.getItem(themeKey) || (choice("D07")?.includes("operating-system") && matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"); } catch (_) {}
setTheme(initialTheme === "dark" ? "dark" : "light");
$$(".theme-switch button").forEach(b => b.addEventListener("click", () => setTheme(b.dataset.theme)));

function showWorkspace(name) {
  $$(".workspace-pane").forEach(p => p.hidden = p.id !== `${name}-pane`);
  $$(".workspace-nav button").forEach(b => { const active = b.dataset.workspace === name; b.classList.toggle("active", active); if (active) b.setAttribute("aria-current", "page"); else b.removeAttribute("aria-current"); });
}
$$(".workspace-nav button").forEach(b => b.addEventListener("click", () => showWorkspace(b.dataset.workspace)));
$("#brand").addEventListener("click", () => {
  if (choice("D03")?.includes("GitHub")) window.open("https://github.com/turbo-sim/turbodash", "_blank", "noopener,noreferrer");
  else if (choice("D03")?.includes("About")) { openDecisions(); $("#decision-D03").scrollIntoView({block:"center"}); }
  else showWorkspace("meanline");
});
$("#collapse-all").addEventListener("click", () => $$(".inspector details").forEach(d => d.open = false));
$$(".file-action,#start-optimization").forEach(b => b.addEventListener("click", () => toast("Review placeholder — the production action will remain in the Dash app.")));

// Stable stage/row state demonstrates selection without rounding or dropping edits.
const aero = [
  ["work_fraction_split","Work split weight","1","–","work distribution fraction"],
  ["stator_exit_angle","Stator exit angle","70","deg","alpha angle"],
  ["degree_reaction","Degree of reaction","0.5","–","reaction R"]
];
const ratios = [
  ["meridional_velocity_ratio_12","Meridional velocity ratio, station 1 / 2","1","–","velocity ratio vm1 vm2"],
  ["meridional_velocity_ratio_23","Meridional velocity ratio, station 2 / 3","1","–","velocity ratio vm2 vm3"],
  ["meridional_velocity_ratio_34","Meridional velocity ratio, station 3 / 4","1","–","velocity ratio vm3 vm4"],
  ...["01","12","23","34"].map(p => [`radius_ratio_${p}`,`Radius ratio, station ${p[0]} / ${p[1]}`,"1","–","radius ratio"])
];
const shape = [
  ["solidity_chord","Solidity, true chord / pitch","1.121","–","solidity chord pitch"],
  ["aspect_ratio","Aspect ratio","2","–","aspect span chord"],
  ["maximum_thickness_to_chord","Maximum thickness / chord","0.3","–","maximum thickness"],
  ["maximum_thickness_location","Maximum thickness location / chord","0.3","–","maximum thickness location"]
];
const edges = [
  ["leading_edge_radius_to_max_thickness","Leading edge radius / maximum thickness","0.5","–","leading edge radius"],
  ["trailing_edge_thickness_to_opening","Trailing edge thickness / opening","0.07","–","trailing edge thickness opening"],
  ["leading_edge_wedge_angle","Leading edge wedge angle","30","deg","leading edge wedge angle"],
  ["trailing_edge_wedge_angle","Trailing edge wedge angle","5","deg","trailing edge wedge angle"]
];
const defaults = fields => Object.fromEntries(fields.map(f => [f[0], f[2]]));
const stages = ["High-pressure stage","Low-pressure stage"].map((name,i) => ({id:`stage-${i+1}`,name, aero:defaults([...aero,...ratios]),
  rows:{stator:defaults([...shape,...edges]),rotor:{...defaults([...shape,...edges]),solidity_chord:"1.137"}}}));
let selectedStage = 0, selectedRow = "stator", stageCount = 1;
function fieldElement(f, rowScoped) {
  const [key, title,, unit] = f;
  const label = document.createElement("label"); label.className = "field"; label.append(document.createTextNode(title));
  const wrapper = document.createElement("span"); wrapper.className = "numeric";
  const input = document.createElement("input"); input.type = "number"; input.step = "any";
  input.id = `field-${key}`; input.dataset.key = key; input.dataset.rowScoped = rowScoped ? "yes" : "no";
  input.value = (rowScoped ? stages[selectedStage].rows[selectedRow] : stages[selectedStage].aero)[key];
  input.addEventListener("input", () => { (rowScoped ? stages[selectedStage].rows[selectedRow] : stages[selectedStage].aero)[key] = input.value; });
  wrapper.append(input); const suffix = document.createElement("span"); suffix.textContent = unit; wrapper.append(suffix); label.append(wrapper);
  return label;
}
function renderStage() {
  for (const [selector, fields, rowScoped] of [["#stage-fields",aero,false],["#ratio-fields",ratios,false],["#blade-fields",shape,true],["#edge-fields",edges,true]]) {
    const container = $(selector); container.replaceChildren(...fields.map(f => fieldElement(f,rowScoped)));
  }
  $("#stage-select").value = String(selectedStage);
  $$(".row-switch button").forEach(b => { const active = b.dataset.row === selectedRow; b.classList.toggle("active",active); b.setAttribute("aria-pressed",String(active)); });
  $("#stage-summary").textContent = `${stages[selectedStage].name} · ${selectedRow === "stator" ? "Stator" : "Rotor"}`;
}
$("#stage-select").addEventListener("change", e => { selectedStage = Number(e.target.value); renderStage(); });
$$(".row-switch button").forEach(b => b.addEventListener("click", () => { selectedRow = b.dataset.row; renderStage(); }));
function setStageCount(count) {
  stageCount = Math.max(1,Math.min(2,Number(count))); selectedStage = Math.min(selectedStage,stageCount-1);
  $("#stage-select").replaceChildren(...stages.slice(0,stageCount).map((s,i) => {const o=document.createElement("option"); o.value=i; o.textContent=s.name; return o;}));
  $("#stage-count").value = String(stageCount); $("#example-count").value = String(stageCount);
  $("#setup-summary").textContent = `Axial · ${stageCount} stage${stageCount>1?"s":""} · ${$("#loss-model").value}`;
  $("#case-name").textContent = stageCount === 1 ? "Axial · Single stage" : "Axial · Two stages";
  renderStage(); renderPlots(); renderTable();
}
$("#stage-count").addEventListener("change",e=>setStageCount(e.target.value));
$("#example-count").addEventListener("change",e=>setStageCount(e.target.value));
$("#loss-model").addEventListener("change",()=>$("#setup-summary").textContent=`Axial · ${stageCount} stage${stageCount>1?"s":""} · ${$("#loss-model").value}`);

const propertyPairs = {PT_INPUTS:[["Inlet pressure","Pa"],["Inlet temperature","K"]],HmassP_INPUTS:[["Inlet enthalpy","J/kg"],["Inlet pressure","Pa"]],PSmass_INPUTS:[["Inlet pressure","Pa"],["Inlet entropy","J/(kg K)"]],PQ_INPUTS:[["Inlet pressure","Pa"],["Inlet vapor quality","kg/kg"]]};
$("#property-pair").addEventListener("change",()=>{
  propertyPairs[$("#property-pair").value].forEach(([label,unit],i)=>{ $("#property-"+(i+1)+"-label").textContent=label; $("#property-"+(i+1)+"-unit").textContent=unit; $("#property-"+(i+1)).value=""; });
  $("#pair-message").hidden=false; $("#operating-summary").textContent=`${$("#fluid").value} · inlet values needed`;
});
for(const id of ["property-1","property-2"]) $("#"+id).addEventListener("input",()=>{ const valid=["property-1","property-2"].every(x=>$("#"+x).value!=="" && Number.isFinite($("#"+x).valueAsNumber)); $("#pair-message").hidden=valid; if(valid)$("#operating-summary").textContent=`${$("#fluid").value} · ${$("#property-1").value} ${$("#property-1-unit").textContent} · ${$("#property-2").value} ${$("#property-2-unit").textContent}`; });
const speedValues={specific:"0.7",actual:"30000"}; let previousSpeedMode="specific";
$("#speed-mode").addEventListener("change",()=>{speedValues[previousSpeedMode]=$("#speed").value; previousSpeedMode=$("#speed-mode").value; $("#speed").value=speedValues[previousSpeedMode]; const actual=previousSpeedMode==="actual"; $("#speed-label").textContent=actual?"Rotational speed":"Specific speed"; $("#speed-unit").textContent=actual?"rpm":"–"; $("#shaft-summary").textContent=`${$("#speed-label").textContent} ${$("#speed").value}${actual?" rpm":""}`;});

// Typeahead navigates context only. It never changes a model parameter.
const commonCatalog = [
  ["turbine-type","Turbine type","setup","axial radial"],["stage-count","Number of stages","setup","count stages"],["loss-model","Loss model","setup","benner kacker okapuu moustapha isentropic"],
  ["fluid","Working fluid","operating","air co2 carbon dioxide water nitrogen refrigerant"],["property-pair","Inlet property pair","operating","pressure temperature enthalpy entropy vapor quality thermodynamic"],
  ["property-1","Inlet property 1","operating",""],["property-2","Inlet property 2","operating",""],
  ["exit-pressure","Exit pressure","operating","outlet pressure"],["mass-flow","Mass flow rate","operating","mdot massflow"],["inlet-angle","Inlet flow angle","operating","alpha inlet"],
  ["speed-mode","Speed definition","shaft","specific actual rpm"],["speed","Shaft speed","shaft","angular rotational specific rpm"],["velocity-ratio","Blade velocity ratio","shaft","nu velocity ratio"],
  ["objective","Optimization objective","optimization","efficiency power maximize minimize"],["lower-bound","Lower bound","optimization","variable lower bound"],["initial-value","Initial value","optimization","initial guess"],["upper-bound","Upper bound","optimization","variable upper bound"],["solver-tolerance","Solver tolerance","optimization","convergence tolerance slsqp"],["iteration-limit","Iteration limit","optimization","max iterations"]
];
function catalog() {
  const result=commonCatalog.map(([id,label,section,aliases])=>({id,label:id.startsWith("property-")&&id!=="property-pair"?$("#"+id+"-label").textContent:label,section,aliases,path:({setup:"Turbine setup",operating:"Operating point",shaft:"Shaft speed",optimization:"Optimization"})[section],subsection:["solver-tolerance","iteration-limit"].includes(id)?"solver-settings":null}));
  stages.slice(0,stageCount).forEach((stage,i)=>{
    [...aero,...ratios].forEach(f=>result.push({id:`field-${f[0]}`,label:f[1],section:"stage",stage:i,aliases:f[4],path:`Stage ${i+1} · ${stage.name}`,subsection:ratios.includes(f)?"stage-ratios":null}));
    for(const row of ["stator","rotor"]) [...shape,...edges].forEach(f=>result.push({id:`field-${f[0]}`,label:f[1],section:"stage",stage:i,row,aliases:f[4],path:`Stage ${i+1} · ${stage.name} · ${row}`,subsection:edges.includes(f)?"stage-edges":null}));
  }); return result;
}
function normalize(text){return text.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g,"").replace(/[-–—/]/g," ");}
let searchMatches=[], activeResult=0;
function closeSearch(){ $("#search-results").hidden=true; $("#parameter-search").setAttribute("aria-expanded","false"); $("#parameter-search").removeAttribute("aria-activedescendant"); }
function selectSearch(index){
  const result=searchMatches[index]; if(!result)return;
  showWorkspace("meanline");
  if(result.stage!==undefined){selectedStage=result.stage;if(result.row)selectedRow=result.row;renderStage();}
  $("#"+result.section).open=true;if(result.subsection)$("#"+result.subsection).open=true;
  closeSearch();
  const field=$("#"+result.id); field.scrollIntoView({behavior:"instant",block:"center"}); field.focus({preventScroll:true}); field.classList.add("found-field"); setTimeout(()=>field.classList.remove("found-field"),2200);
}
function updateActive(){
  $$(".search-result").forEach((r,i)=>r.setAttribute("aria-selected",String(i===activeResult)));
  if(searchMatches.length)$("#parameter-search").setAttribute("aria-activedescendant",`search-option-${activeResult}`);
}
function search(){
  const tokens=normalize($("#parameter-search").value.trim()).split(/\s+/).filter(Boolean);if(!tokens.length){closeSearch();return;}
  searchMatches=catalog().filter(r=>tokens.every(t=>normalize(`${r.label} ${r.aliases} ${r.path}`).includes(t))).slice(0,8);activeResult=0;
  const list=$("#search-results");list.replaceChildren();
  if(!searchMatches.length){const empty=document.createElement("p");empty.className="search-empty";empty.textContent="No matching parameter. Try a name, stage or blade row.";list.append(empty);}
  searchMatches.forEach((r,i)=>{const item=document.createElement("div");item.className="search-result";item.role="option";item.id=`search-option-${i}`;item.setAttribute("aria-selected",String(i===0));const title=document.createElement("strong");title.textContent=r.label;const path=document.createElement("small");path.textContent=r.path;item.append(title,path);item.addEventListener("mousedown",e=>e.preventDefault());item.addEventListener("click",()=>selectSearch(i));list.append(item);});
  list.hidden=false;$("#parameter-search").setAttribute("aria-expanded","true");updateActive();
}
$("#parameter-search").addEventListener("input",search);
$("#parameter-search").addEventListener("keydown",e=>{
  if(e.key==="Escape"){closeSearch();e.preventDefault();}
  else if(["ArrowDown","ArrowUp"].includes(e.key)&&searchMatches.length){e.preventDefault();activeResult=(activeResult+(e.key==="ArrowDown"?1:-1)+searchMatches.length)%searchMatches.length;updateActive();$("#search-option-"+activeResult).scrollIntoView({block:"nearest"});}
  else if(e.key==="Enter"&&!$("#search-results").hidden){e.preventDefault();selectSearch(activeResult);}
});
document.addEventListener("keydown",e=>{if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==="k"){e.preventDefault();$("#parameter-search").focus();$("#parameter-search").select();}});
document.addEventListener("click",e=>{if(!e.target.closest(".search-wrap"))closeSearch();});
$("#parameter-search").addEventListener("blur",()=>setTimeout(closeSearch,120));

// Deliberately schematic, ordered station geometry. Production must use the solved geometry.
function renderPlots(){
  const xmax=stageCount*160+20, x=v=>65+v/xmax*520, y=v=>290-v/260*245;
  let grid="",ticks="";
  for(const r of [0,50,100,150,200,250]){grid+=`<path class="grid" d="M65 ${y(r)}H585"/>`;ticks+=`<text x="53" y="${y(r)+4}" text-anchor="end">${r}</text>`;}
  for(let i=0;i<=4;i++){const v=xmax*i/4;grid+=`<path class="grid" d="M${x(v)} 45V290"/>`;ticks+=`<text x="${x(v)}" y="309" text-anchor="middle">${Number(v.toFixed(1))}</text>`;}
  const rows=[];
  for(let s=0;s<stageCount;s++){rows.push({x0:s*160+15,x1:s*160+65,h0:150-s*5,h1:148-s*5,s0:205+s*7,s1:211+s*7,kind:"stator",stage:s+1});rows.push({x0:s*160+83,x1:s*160+145,h0:148-s*5,h1:145-s*5,s0:211+s*7,s1:218+s*7,kind:"rotor",stage:s+1});}
  const stations=[{x:0,h:rows[0].h0,s:rows[0].s0},...rows.flatMap(r=>[{x:r.x0,h:r.h0,s:r.s0},{x:r.x1,h:r.h1,s:r.s1}]),{x:xmax,h:rows.at(-1).h1,s:rows.at(-1).s1}];
  const path=k=>stations.map((p,i)=>`${i?"L":"M"}${x(p.x)} ${y(p[k])}`).join(" ");
  let shapes=`<path class="hub-solid" d="${path("h")}L${x(xmax)} ${y(0)}L${x(0)} ${y(0)}Z"/><path class="hub-solid" d="${path("s")}L${x(xmax)} ${y(260)}L${x(0)} ${y(260)}Z"/><path class="wall-line hub" d="${path("h")}"/><path class="wall-line shroud" d="${path("s")}"/>`;
  for(const r of rows){shapes+=`<path class="${r.kind==="stator"?"passage":"rotor-passage"}" d="M${x(r.x0)} ${y(r.h0)}L${x(r.x1)} ${y(r.h1)}L${x(r.x1)} ${y(r.s1)}L${x(r.x0)} ${y(r.s0)}Z"/><text x="${x((r.x0+r.x1)/2)}" y="${y((r.h0+r.s0)/2)+4}" text-anchor="middle">${r.kind==="stator"?"S":"R"}${r.stage}</text>`;}
  let arrows="";for(let s=0;s<stageCount;s++){const start=x(s*160+68),end=x(s*160+79),ym=y(180+s);arrows+=`<path class="arrow" d="M${start} ${ym}L${end} ${ym-1}m-5-4 5 4-4 5"/>`;}
  const wallLabels=`<text x="${x(xmax)-3}" y="${y(rows.at(-1).s1)-9}" text-anchor="end">Shroud</text><text x="${x(xmax)-3}" y="${y(rows.at(-1).h1)+17}" text-anchor="end">Hub</text>`;
  $("#meridional").innerHTML=`<svg viewBox="0 0 630 345" aria-label="${stageCount}-stage schematic meridional channel with hub and shroud, radial axis starts at zero" role="img">${grid}${shapes}${arrows}${wallLabels}<path class="axis" d="M65 35V290H585"/>${ticks}<text x="325" y="336" text-anchor="middle">Axial direction [mm]</text><text transform="translate(18 170) rotate(-90)" text-anchor="middle">Radius [mm]</text></svg>`;
  let blades="";const groups=stageCount*2;for(let g=0;g<groups;g++){const cx=75+(g+.5)*490/groups,scale=stageCount===1?1:.64;for(let r=0;r<3;r++){const yy=65+r*66;blades+=`<g transform="translate(${cx} ${yy}) scale(${scale} 1)"><path class="blade-line ${g%2?"rotor":""}" d="${g%2?"M-40 2C-15-15 20-5 50 43C20 17-5 22-35 18Q-53 13-40 2Z":"M-45 33C-18 35 15-7 50-8C18 3-10 49-42 45Q-55 41-45 33Z"}"/></g>`;}blades+=`<text x="${cx}" y="277" text-anchor="middle">${g%2?"R":"S"}${Math.floor(g/2)+1}</text>`;}
  $("#blade-plot").innerHTML=`<svg viewBox="0 0 630 345" role="img" aria-label="Illustrative blade passages for ${stageCount} stages">${[80,145,210,270].map(yy=>`<path class="grid" d="M65 ${yy}H585"/>`).join("")}${[80,180,280,380,480,580].map(xx=>`<path class="grid" d="M${xx} 40V290"/>`).join("")}${blades}<path class="axis" d="M65 35V290H585"/><text x="325" y="336" text-anchor="middle">Axial direction [mm]</text><text transform="translate(18 170) rotate(-90)" text-anchor="middle">Tangential direction [mm]</text><text x="325" y="310" text-anchor="middle">Blade row</text></svg>`;
}
function formatPower(value){
  if(typeof value!=="number"||!Number.isFinite(value))return{value:"—",unit:""};
  const magnitude=Math.abs(value),scales=[1,1000,1000000],units=["W","kW","MW"];
  let i=magnitude>=1000000?2:magnitude>=1000?1:0;
  if(i<2&&Math.round((magnitude/scales[i]+Number.EPSILON)*100)/100>=1000)i++;
  return{value:(value/scales[i]).toFixed(2),unit:units[i]};
}
let activeReport="performance";
function renderTable(){
  let headers,rows;
  if(activeReport==="performance"){
    headers=["Performance metric","Unit","Turbine",...stages.slice(0,stageCount).map(s=>s.name)];
    rows=[["Total-to-static efficiency","%","84.50","84.50","85.20"],["Total-to-total efficiency","%","90.10","90.10","90.40"],["Rotational speed","rpm","30000","30000","30000"],["Power","MW","1.35",stageCount===1?"1.35":"0.72","0.63"],["Pressure ratio","–","5.00",stageCount===1?"5.00":"2.25","2.22"]].map(r=>r.slice(0,3+stageCount));
  }else{
    headers=[activeReport==="geometry"?"Geometry metric":"Flow property","Unit",...Array.from({length:stageCount*2},(_,i)=>`${i%2?"Rotor":"Stator"} ${Math.floor(i/2)+1}`)];
    rows=(activeReport==="geometry"?[["True-chord solidity","–","1.121","1.137","1.121","1.137"],["Aspect ratio","–","2.000","2.000","2.000","2.000"],["Maximum thickness / chord","–","0.300","0.300","0.300","0.300"],["Trailing-edge thickness / opening","–","0.070","0.070","0.070","0.070"]]:[["Exit static pressure","Pa","320000","100000","180000","100000"],["Exit static temperature","K","548.00","475.00","498.00","465.00"],["Exit Mach number","–","0.82","0.68","0.84","0.70"],["Mass flow rate","kg/s","10.00","10.00","10.00","10.00"]]).map(r=>r.slice(0,2+stageCount*2));
  }
  const table=$("#report-table");table.replaceChildren();const head=document.createElement("thead"),tr=document.createElement("tr");headers.forEach(h=>{const th=document.createElement("th");th.scope="col";th.textContent=h;tr.append(th);});head.append(tr);table.append(head);const body=document.createElement("tbody");rows.forEach(r=>{const row=document.createElement("tr");r.forEach(t=>{const td=document.createElement("td");td.textContent=t;row.append(td);});body.append(row);});table.append(body);
}
$$(".report-tabs button").forEach(b=>b.addEventListener("click",()=>{activeReport=b.dataset.report;$$(".report-tabs button").forEach(x=>x.classList.toggle("active",x===b));renderTable();}));

function openDecisions(){ $("#decisions-dialog").showModal(); }
$("#open-decisions").addEventListener("click",openDecisions);
$("#close-decisions").addEventListener("click",()=>$("#decisions-dialog").close());
$$("[data-feedback]").forEach(b=>b.addEventListener("click",()=>{openDecisions();$("#general-feedback").focus();}));
for(const d of decisions.decisions){
  const card=document.createElement("section");card.className="decision";card.id=`decision-${d.id}`;
  const title=document.createElement("h3"),id=document.createElement("span");id.textContent=d.id;title.append(id,document.createTextNode(d.title));
  const reason=document.createElement("p");reason.textContent=d.reason;const recommendation=document.createElement("p");recommendation.className="recommendation";recommendation.textContent=`Recommended: ${d.recommended}`;
  const token=document.createElement("code");token.textContent=d.placeholder;
  const selectLabel=document.createElement("label");selectLabel.textContent="Your choice";const select=document.createElement("select");select.id=`choice-${d.id}`;
  const pending=document.createElement("option");pending.value="";pending.textContent="Not decided — leave as a placeholder";select.append(pending);d.options.forEach(text=>{const o=document.createElement("option");o.value=text;o.textContent=text;select.append(o);});select.value=d.choice||"";
  select.addEventListener("change",()=>{d.choice=select.value||null;saveFeedback();updateBrand();});selectLabel.append(select);
  const notesLabel=document.createElement("label");notesLabel.textContent="Your notes or another option";const notes=document.createElement("textarea");notes.id=`notes-${d.id}`;notes.rows=2;notes.value=d.notes;notes.placeholder="Edit this suggestion or describe your preference…";notes.addEventListener("input",()=>{d.notes=notes.value;saveFeedback();});notesLabel.append(notes);
  card.append(title,reason,recommendation,token,selectLabel,notesLabel);$("#decision-fields").append(card);
}
$("#general-feedback").value=generalFeedback;
$("#general-feedback").addEventListener("input",e=>{generalFeedback=e.target.value;saveFeedback();});
$("#export-feedback").addEventListener("click",()=>{
  const payload=saveFeedback(),blob=new Blob([JSON.stringify(payload,null,2)],{type:"application/json"});const url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download="turbodash-design-feedback.json";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
$("#pending-count").textContent=decisions.decisions.filter(d=>!d.choice).length;
setStageCount(1);
const power=formatPower(1350000);$("#power-kpi").innerHTML=`${power.value} <small>${power.unit}</small>`;
window.reviewAPI={formatPower,getStages:()=>structuredClone(stages),setStageCount,setTheme};
