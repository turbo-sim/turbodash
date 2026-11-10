/* Three.js scene for sampled impeller reports. Navigation reuses the original
 * TurboMoC orientation overlay through its existing camera interface. */
import * as THREE from 'impeller/three';
import {TrackballControls} from 'impeller/trackball';

const host = document.getElementById('impeller-viewer');
const status = document.getElementById('viewer-status');
let data = window.impellerReport || null;

function start() {
  if (!host) throw new Error('Missing impeller viewer container.');
  host.classList.add('td-vtk-host');
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(36, 1, .00001, 100);
  camera.up.set(0, 0, 1);
  const renderer = new THREE.WebGLRenderer({antialias:true, alpha:true, preserveDrawingBuffer:true});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0xffffff, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.35;
  host.appendChild(renderer.domElement);
  renderer.domElement.tabIndex = 0;
  renderer.domElement.setAttribute('aria-label', 'Impeller model. Drag to rotate, right-drag to pan, scroll to zoom.');

  const controls = new TrackballControls(camera, renderer.domElement);
  controls.staticMoving = true;
  controls.rotateSpeed = 3;
  controls.zoomSpeed = 1.1;
  controls.panSpeed = .5;
  let model = new THREE.Group();
  scene.add(model);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x6c8590, 2.1));
  const lightGroup = new THREE.Group();
  camera.add(lightGroup); scene.add(camera);
  for (const [position, intensity] of [[[2,3,4],3.5],[[-3,-1,2],1.7],[[0,-3,-3],1.3]]) {
    const light = new THREE.DirectionalLight(0xffffff, intensity);
    light.position.set(...position); lightGroup.add(light);
  }
  function geometry(mesh) {
    let vertices = mesh.vertices, faces = mesh.faces;
    const angularSamples = mesh.angular_samples, hubSamples = mesh.hub_profile_samples;
    if(Array.isArray(mesh.face_regions) && mesh.face_regions.length === faces.length) {
      // Geometry supplies regions for arbitrary profiles, including axis poles
      // and rounded noses. Only the display mesh is split at sharp corners.
      const displayVertices=[], displayFaces=[], indices=new Map();
      faces.forEach((face, faceIndex)=>{
        const region=mesh.face_regions[faceIndex];
        displayFaces.push(face.map(index=>{
          const key=`${region}:${index}`;
          if(!indices.has(key)) {indices.set(key,displayVertices.length);displayVertices.push(vertices[index]);}
          return indices.get(key);
        }));
      });
      vertices=displayVertices;faces=displayFaces;
    } else if(Number.isInteger(angularSamples) && Number.isInteger(hubSamples)
      && angularSamples >= 3 && hubSamples >= 2
      && vertices.length === angularSamples*(hubSamples+3)) {
      // The disk export is a welded surface of revolution. Keep that topology
      // in the export, but separate its five display regions so normal averaging
      // does not round the front/back annuli or the bore/rim corners.
      const displayVertices=[], displayFaces=[], indices=new Map();
      for(const face of faces) {
        const rings=face.map(index=>Math.floor(index/angularSamples));
        const lower=Math.min(...rings), upper=Math.max(...rings);
        const strip=lower===0 && upper===hubSamples+2 ? upper : lower;
        const region=strip<hubSamples-1 ? 0 : strip-hubSamples+2;
        displayFaces.push(face.map(index=>{
          const key=`${region}:${index}`;
          if(!indices.has(key)) {indices.set(key,displayVertices.length);displayVertices.push(vertices[index]);}
          return indices.get(key);
        }));
      }
      vertices=displayVertices;faces=displayFaces;
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices.flat(), 3));
    geometry.setIndex(faces.flat());
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();
    return geometry;
  }
  let bladeGroup = new THREE.Group(), edgeGroup = new THREE.Group();
  let disk = null, shroud = null, count = 0, radius = .3, hasGeometry = false;
  const center = new THREE.Vector3();
  const homeDirection = new THREE.Vector3(1,-1,-1.25).normalize();
  camera.position.copy(homeDirection).multiplyScalar(1.4);
  camera.lookAt(center);

  function disposeModel(group) {
    const geometries = new Set(), materials = new Set();
    group.traverse(object => {
      if(object.geometry) geometries.add(object.geometry);
      if(object.material) for(const material of [].concat(object.material)) materials.add(material);
    });
    geometries.forEach(item=>item.dispose());
    materials.forEach(item=>item.dispose());
  }

  function buildModel(next) {
    if(!next?.blade || !Number.isInteger(next.blade_count) || next.blade_count < 1)
      throw new Error('The viewer needs one blade and a positive integer blade count.');
    const replacement = new THREE.Group();
    const blades = new THREE.Group(), edges = new THREE.Group();
    replacement.add(blades,edges);
    try {
      const bladeGeometry = geometry(next.blade);
      const bladeMaterial = new THREE.MeshStandardMaterial({color:0x438d9e,metalness:.42,roughness:.34,side:THREE.DoubleSide});
      const edgeTemplates = [];
      for(const [name,color] of [['leading',0xd3933a],['trailing',0x314f69]]) {
        const points = next.edges?.[name];
        if(points?.length > 1) edgeTemplates.push({
          geometry:new THREE.BufferGeometry().setFromPoints(points.map(point=>new THREE.Vector3(...point))),
          material:new THREE.LineBasicMaterial({color,depthTest:true})
        });
      }
      for(let index=0;index<next.blade_count;index++) {
        const angle = 2*Math.PI*index/next.blade_count;
        const blade = new THREE.Mesh(bladeGeometry,bladeMaterial);
        blade.rotation.z=angle;blades.add(blade);
        const edge = new THREE.Group();edge.rotation.z=angle;edges.add(edge);
        for(const item of edgeTemplates) edge.add(new THREE.Line(item.geometry,item.material));
      }
      let nextDisk=null,nextShroud=null;
      if(next.disk) {
        nextDisk=new THREE.Mesh(geometry(next.disk),new THREE.MeshStandardMaterial({color:0xa7b4be,metalness:.55,roughness:.4,side:THREE.DoubleSide}));
        replacement.add(nextDisk);
      }
      if(next.shroud) {
        nextShroud=new THREE.Mesh(geometry(next.shroud),new THREE.MeshStandardMaterial({color:0x7da2b2,metalness:.05,roughness:.7,side:THREE.DoubleSide,transparent:true,opacity:.16,depthWrite:false}));
        replacement.add(nextShroud);
      }
      // A tight rotational sphere encloses all repeated blades without a row payload.
      const meshes=[next.blade,next.disk,next.shroud].filter(Boolean);
      let minimumZ=Infinity,maximumZ=-Infinity;
      for(const mesh of meshes) for(const point of mesh.vertices) {
        minimumZ=Math.min(minimumZ,point[2]);maximumZ=Math.max(maximumZ,point[2]);
      }
      const nextCenter=new THREE.Vector3(0,0,(minimumZ+maximumZ)/2);
      let radiusSquared=0;
      for(const mesh of meshes) for(const [x,y,z] of mesh.vertices)
        radiusSquared=Math.max(radiusSquared,x*x+y*y+(z-nextCenter.z)**2);
      if(!Number.isFinite(radiusSquared) || radiusSquared<=0) throw new Error('Geometry bounds are invalid.');
      return {model:replacement,blades,edges,disk:nextDisk,shroud:nextShroud,center:nextCenter,radius:Math.sqrt(radiusSquared)};
    } catch(error) {disposeModel(replacement);throw error;}
  }

  function updateGeometry(next, options={}) {
    // Construct first: malformed replacement data must not clear the accepted view.
    const replacement=buildModel(next);
    const old=model;
    model=replacement.model;bladeGroup=replacement.blades;edgeGroup=replacement.edges;
    disk=replacement.disk;shroud=replacement.shroud;
    center.copy(replacement.center);radius=replacement.radius;count=next.blade_count;
    scene.add(model);scene.remove(old);disposeModel(old);
    data=next;window.impellerReport=next;
    controls.minDistance=radius*.15;controls.maxDistance=radius*30;
    camera.near=radius/2000;camera.far=radius*100;camera.updateProjectionMatrix();
    // User orbit, pan, zoom and display controls survive ordinary geometry edits.
    if(!hasGeometry || options.resetCamera) fit(true);
    hasGeometry=true;
    updateVisibility();
    host.dataset.fingerprint=String(next.fingerprint || '');
    host.dataset.ready='true';
    if(status) status.textContent=`Geometry loaded · ${count} main blades`;
    return {fingerprint:next.fingerprint,blade_count:count};
  }

  let cube, frame = null;
  const cameraListeners = new Set();
  const notify = () => cameraListeners.forEach(callback=>callback());
  function draw() { renderer.render(scene,camera); }
  function requestDraw() {
    if (frame !== null) return;
    frame = requestAnimationFrame(()=>{frame=null;controls.update();draw();});
  }
  controls.addEventListener('change', ()=>{notify();requestDraw();});
  // TrackballControls applies its accumulated pointer/wheel deltas on update;
  // event-driven frames keep the model idle when nobody is interacting.
  for (const event of ['pointerdown','pointermove','pointerup','wheel','touchmove'])
    renderer.domElement.addEventListener(event, requestDraw, {passive:true});
  document.addEventListener('pointermove', event=>{if(event.buttons)requestDraw();}, {passive:true});
  document.addEventListener('pointerup', requestDraw, {passive:true});
  function fit(resetOrientation=true) {
    controls.target.copy(center);
    const vertical = THREE.MathUtils.degToRad(camera.fov)/2;
    const horizontal = Math.atan(Math.tan(vertical)*camera.aspect);
    const distance = radius/Math.sin(Math.min(vertical,horizontal))*1.10;
    const direction = resetOrientation ? homeDirection : camera.position.clone().sub(controls.target).normalize();
    camera.position.copy(center).addScaledVector(direction,distance);
    if(resetOrientation) camera.up.set(0,0,1);
    camera.lookAt(controls.target); controls.update();notify();requestDraw();
  }
  const adapter = {
    camera: {
      getPosition:()=>camera.position.toArray(),getFocalPoint:()=>controls.target.toArray(),getViewUp:()=>camera.up.toArray(),
      setPosition:(...values)=>{camera.position.set(...values);notify();},
      setViewUp:(...values)=>{camera.up.set(...values);notify();},
      onModified:callback=>{cameraListeners.add(callback);return {unsubscribe:()=>cameraListeners.delete(callback)};},
    },
    style:{setCenterOfRotation:()=>{}},
    renderer:{resetCameraClippingRange:()=>{}},
    renderWindow:{render:()=>{camera.lookAt(controls.target);controls.update();draw();}},
    containerRef:{current:renderer.domElement},
  };
  cube = window.turbomocOrientation.create(host,'impeller');
  cube.attach(adapter);cube.setHome(homeDirection.toArray(),[0,0,1]);cube.setEnabled(true);

  function updateVisibility() {
    const single = document.getElementById('blade-mode')?.value === 'single';
    bladeGroup.children.forEach((mesh,index)=>mesh.visible=!single || index===0);
    edgeGroup.children.forEach((edge,index)=>edge.visible=!single || index===0);
    edgeGroup.visible = document.getElementById('show-edges')?.checked ?? true;
    if(disk) disk.visible = document.getElementById('show-disk')?.checked ?? true;
    if(shroud) shroud.visible = document.getElementById('show-shroud')?.checked ?? false;
    if(disk) {
      let value = Number(document.getElementById('disk-opacity')?.value ?? 1);
      if(value>1) value/=100;
      disk.material.opacity = Math.min(1,Math.max(0,value));
      disk.material.transparent = disk.material.opacity < 1;
      disk.material.depthWrite = !disk.material.transparent;
      disk.material.needsUpdate = true;
    }
    requestDraw();
  }
  for(const id of ['blade-mode','show-disk','show-shroud','show-edges','disk-opacity'])
    document.getElementById(id)?.addEventListener('input',updateVisibility);
  document.getElementById('reset-view')?.addEventListener('click',()=>fit(true));
  const tools = document.createElement('div');tools.className='viewer-tools';
  const fitButton=document.createElement('button');fitButton.textContent='Fit';fitButton.title='Fit the complete impeller in the viewport';fitButton.addEventListener('click',()=>fit(false));
  const existingFullButton=document.getElementById('fullscreen-view');
  const fullButton=existingFullButton || document.createElement('button');
  if(!existingFullButton)fullButton.textContent='Full screen';
  fullButton.addEventListener('click',async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await (host.closest('.viewport-wrap') || host).requestFullscreen();}catch(error){if(status)status.textContent=error.message;}});
  tools.append(fitButton);if(!existingFullButton)tools.append(fullButton);host.append(tools);
  if(!host.closest('.viewport-wrap')){
    const hint=document.createElement('div');hint.className='viewer-hint';hint.textContent='Drag to rotate · scroll to zoom\nRight-drag to pan · cube to align';hint.style.whiteSpace='pre-line';host.append(hint);
  }
  let firstSize=true;
  const resize = ()=>{
    const width=Math.max(1,host.clientWidth),height=Math.max(1,host.clientHeight);
    camera.aspect=width/height;camera.updateProjectionMatrix();renderer.setSize(width,height,false);controls.handleResize();
    if(firstSize){firstSize=false;if(hasGeometry)fit(true);else requestDraw();}else requestDraw();
  };
  new ResizeObserver(resize).observe(host);resize();updateVisibility();
  renderer.domElement.addEventListener('webglcontextlost',event=>{event.preventDefault();if(status)status.textContent='The graphics context was lost. Reload the report to restore the model.';});
  window.impellerViewer = {camera,controls,scene,renderer,cube,fit,draw:requestDraw,updateGeometry,
    get bladeGroup(){return bladeGroup;},get disk(){return disk;},get shroud(){return shroud;},get edgeGroup(){return edgeGroup;}};
  if(data) updateGeometry(data);
  else {host.dataset.ready='waiting';if(status)status.textContent='Waiting for the first valid geometry…';}
  window.dispatchEvent(new CustomEvent('impeller-viewer-ready'));
}

try {start();} catch(error) {
  console.error(error);
  if(status) status.textContent='3D viewer could not start: '+error.message;
  if(host){const message=document.createElement('p');message.className='viewer-failure';message.textContent='The 3D view requires a modern browser with WebGL 2 enabled. '+error.message;host.append(message);host.dataset.ready='error';}
}
