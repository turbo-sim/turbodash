/* Client draft transactions are synchronous. Heavy server results only replace
 * plots and geometry when their revision still equals the latest user edit. */
(function () {
  'use strict';
  const clone = value => JSON.parse(JSON.stringify(value));
  function get(config, path) {
    return path.split('.').reduce((value, key) => Array.isArray(value)
      ? value[Number(key)<0 ? value.length+Number(key) : Number(key)] : value[key], config);
  }
  function put(config, path, value) {
    const keys=path.split('.'),key=keys.pop();
    const parent=keys.length ? get(config, keys.join('.')) : config;
    parent[Array.isArray(parent) && Number(key)<0 ? parent.length+Number(key) : key]=value;
  }
  function display(config,path,spec) {
    let value=get(config,path);
    if(value===null && /^(hub|shroud)\.(inlet|outlet)_angle_deg$/.test(path))
      value=config.dimensions[path.includes('.inlet_') ? 'inlet_flow_angle_deg':'outlet_flow_angle_deg'];
    return value===null ? null : value*spec.scale;
  }
  const help = {
    trailing_edge:'Zero rake aligns the trailing edges. The original turning laws are retained; leading-edge lean is an outcome.',
    free:'Leading-edge stacking and turning laws are prescribed. Exit rake is an outcome.',
    preserve_le:'Leading-edge stacking is retained. Interior monotonic turning changes to satisfy rake; turning location and interpolation are inactive.'
  };
  window.dash_clientside = Object.assign({}, window.dash_clientside, {
    impeller: {
      edit: function(numbers,sliders,mode,interpolation,reset,replacement,
                     draft,revision,baseline,specs,numberIds,sliderIds) {
        const dc=window.dash_clientside, unchanged=Array(13).fill(dc.no_update);
        const triggered=dc.callback_context.triggered_id;
        if(!triggered) return unchanged;
        let next=clone(draft), notice=dc.no_update;
        if(typeof triggered==='object') {
          const ids=triggered.type==='impeller-number' ? numberIds:sliderIds;
          const values=triggered.type==='impeller-number' ? numbers:sliders;
          const index=ids.findIndex(id=>id.path===triggered.path);
          if(index<0 || !specs[triggered.path]) return unchanged;
          const value=values[index];
          if(display(next,triggered.path,specs[triggered.path])===value) return unchanged;
          put(next,triggered.path,value===null ? null : value/specs[triggered.path].scale);
        } else if(triggered==='impeller-rake-mode') {
          if(next.rake.mode===mode) return unchanged;
          next.rake.mode=mode;
          if(mode==='free') {next.rake.angle_deg=0;next.rake.reference_azimuth_deg=null;}
        } else if(triggered==='impeller-angle-interpolation') {
          if(next.sections.every(section=>section.camberline.interpolation===interpolation)) return unchanged;
          for(const section of next.sections) section.camberline.interpolation=interpolation;
        } else if(triggered==='impeller-reset') {
          next=clone(baseline);notice='Example restored, including the original wall control points.';
        } else if(triggered==='impeller-config-replacement') {
          if(!replacement) return unchanged;
          if(replacement.base_revision!==revision) {
            dc.set_props('wall-notice',{children:'Inputs changed while this operation was being checked. Apply or load again.'});
            return unchanged;
          }
          if(replacement.error) {
            dc.set_props('impeller-error',{children:replacement.error});
            dc.set_props('wall-notice',{children:replacement.error});
            return unchanged;
          }
          next=clone(replacement.config);notice=replacement.notice;
        } else return unchanged;
        if(JSON.stringify(next)===JSON.stringify(draft)
          && triggered!=='impeller-reset' && triggered!=='impeller-config-replacement') {
          if(notice!==dc.no_update) dc.set_props('wall-notice',{children:notice});
          return unchanged;
        }
        const currentRevision=(revision || 0)+1;
        window.__impellerDraftRevision=currentRevision;
        dc.set_props('impeller-status',{children:'Updating geometry…'});
        dc.set_props('impeller-error',{children:''});
        dc.set_props('impeller-yaml-download-button',{disabled:true});
        dc.set_props('impeller-stl-download-button',{disabled:true});
        dc.set_props('impeller-stl-status',{children:window.__impellerStlBusy
          ? 'Design changed. The in-progress STL will be discarded; export again when ready.'
          : 'Updating geometry before STL export…'});
        const values=numberIds.map(id=>display(next,id.path,specs[id.path]));
        const sliderValues=sliderIds.map(id=>display(next,id.path,specs[id.path]));
        const disabled=path=>(path==='rake.angle_deg' && next.rake.mode==='free')
          || (path==='sections.-1.stacking_offset_deg' && next.rake.mode==='trailing_edge')
          || (path.includes('camberline.turning_location') && next.rake.mode==='preserve_le')
          || (path==='disk.bore_radius_m' && next.disk.nose_length_m>0);
        return [next,currentRevision,values,sliderValues,
          sliderIds.map((id,i)=>Math.min(specs[id.path].min,sliderValues[i] ?? specs[id.path].min)),
          sliderIds.map((id,i)=>Math.max(specs[id.path].max,sliderValues[i] ?? specs[id.path].max)),
          numberIds.map(id=>disabled(id.path)),sliderIds.map(id=>disabled(id.path)),
          next.rake.mode,next.sections[0].camberline.interpolation,next.rake.mode==='preserve_le',
          help[next.rake.mode],notice];
      },
      accept: function(packet,revision) {
        const dc=window.dash_clientside,skip=Array(10).fill(dc.no_update);
        if(!packet || packet.revision!==revision
          || (window.__impellerDraftRevision!==undefined && packet.revision!==window.__impellerDraftRevision)) return skip;
        if(!packet.valid) {
          dc.set_props('impeller-stl-download-button',{disabled:true});
          if(!window.__impellerStlBusy) dc.set_props('impeller-stl-status',{
            children:'Fix the invalid inputs before exporting STL.'});
          return [dc.no_update,dc.no_update,dc.no_update,dc.no_update,dc.no_update,dc.no_update,
            dc.no_update,'Invalid draft — showing last valid geometry',packet.error,true];
        }
        const accepted={config:packet.config,revision:packet.revision,fingerprint:packet.fingerprint};
        dc.set_props('impeller-stl-download-button',{disabled:!!window.__impellerStlBusy});
        if(!window.__impellerStlBusy) dc.set_props('impeller-stl-status',{
          children:'Full disk and blade row, fused into one solid. STL coordinates are millimetres.'});
        return [packet.figures.angles,packet.figures.thickness,packet.figures.channel,packet.figures.stacking,
          packet.metrics,packet.viewer,accepted,'Geometry up to date',packet.warnings.join(' '),false];
      },
      requestStl: function(clicks,accepted,revision) {
        const dc=window.dash_clientside;
        if(!clicks || window.__impellerStlBusy) return dc.no_update;
        if(!accepted || accepted.revision!==revision
          || (window.__impellerDraftRevision!==undefined && revision!==window.__impellerDraftRevision)) {
          dc.set_props('impeller-stl-status',{children:'Wait for a valid current geometry before exporting STL.'});
          return dc.no_update;
        }
        window.__impellerStlBusy=true;
        window.__impellerStlRequestId=clicks;
        dc.set_props('impeller-stl-download-button',{disabled:true,children:'Preparing STL…'});
        dc.set_props('impeller-stl-status',{children:'Fusing the disk and blades, then checking the saved solid…'});
        return {...clone(accepted),request_id:clicks};
      },
      completeStl: function(packet,accepted,revision) {
        const dc=window.dash_clientside;
        if(!packet || packet.request_id!==window.__impellerStlRequestId) return dc.no_update;
        window.__impellerStlBusy=false;
        const current=accepted && accepted.revision===revision
          && (window.__impellerDraftRevision===undefined || revision===window.__impellerDraftRevision);
        dc.set_props('impeller-stl-download-button',{disabled:!current,children:'Download full STL'});
        if(!current || packet.revision!==revision || packet.fingerprint!==accepted.fingerprint) {
          dc.set_props('impeller-stl-status',{children:'Design changed during export. The old STL was discarded; download the current design again.'});
          return dc.no_update;
        }
        if(!packet.valid) {
          dc.set_props('impeller-stl-status',{children:'STL export failed: '+packet.error});
          return dc.no_update;
        }
        const size=packet.dimensions_mm.map(value=>Number(value).toFixed(2)).join(' × ');
        dc.set_props('impeller-stl-status',{children:'STL ready · '+size+' mm · one fused solid. Check printer fit and supports in your slicer.'});
        return packet.download;
      }
    }
  });
}());
