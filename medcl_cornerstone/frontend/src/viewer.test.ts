import { beforeEach, describe, expect, it, vi } from "vitest";
import { CleanupBag } from "./lifecycle";
import type { ParsedEnvelope } from "./volumeProtocol";

const f = vi.hoisted(() => {
  class Node extends EventTarget {
    children: Node[]=[]; textContent=''; value=''; checked=false; type=''; attributes=new Map();
    classList={toggle:vi.fn()}; style={};
    constructor(public tag='div'){super();}
    append(...nodes:Node[]){this.children.push(...nodes);}
    replaceChildren(...nodes:Node[]){this.children=nodes;}
    setAttribute(k:string,v:string){this.attributes.set(k,v);}
    getContext(){return {};}
  }
  const volumes=new Map<string,any>(), geometries=new Map<string,any>(), states=new Map<string,any>(), reps=new Map<string,any[]>(), luts=new Map<number,any>();
  const engines:any[]=[];
  let failGeometry = false;
  const actor=(kind:string,geometry?:any)=>{
    const property:any={}; const poly={...geometry,delete:vi.fn()}; const mapper={getInputData:()=>poly,delete:vi.fn()};
    const a:any={kind,visible:true,property,geometry,getMapper:()=>mapper,delete:vi.fn(),isA:(x:string)=>x===kind,
      setVisibility:(v:boolean)=>a.visible=v,getVisibility:()=>a.visible,
      getProperty:()=>({setColor:(...v:number[])=>property.color=v,setOpacity:(v:number)=>property.opacity=v,
        setAmbient:(v:number)=>property.ambient=v,setDiffuse:(v:number)=>property.diffuse=v,setSpecular:(v:number)=>property.specular=v})};return a;
  };
  class Engine {
    viewports=new Map<string,any>(); destroy=vi.fn(()=>this.viewports.clear()); resize=vi.fn(); render=vi.fn();
    constructor(){engines.push(this);}
    setViewports(specs:any[]){specs.forEach(s=>{
      const v:any={id:s.viewportId,element:s.element,type:s.type,actors:[],camera:{viewPlaneNormal:[0,0,1]},
        setVolumes:vi.fn(async(inputs:any[])=>{v.actors=inputs.map(i=>({uid:i.actorUID||i.volumeId,actor:actor('vtkVolume'),input:volumes.get(i.volumeId)}));}),
        getActors:()=>v.actors,getActor:(id:string)=>v.actors.find((a:any)=>a.uid===id),
        setProperties:vi.fn(),resetProperties:vi.fn(()=>{if(s.type==='3D'&&v.actors.some((a:any)=>a.actor.kind==='vtkActor'))throw Error('scalar reset on surface');}),
        resetCamera:vi.fn(),setCamera:vi.fn((x:any)=>v.camera=x),getCamera:()=>v.camera,
        getSliceIndex:()=>0,getNumberOfSlices:()=>8,render:vi.fn()};this.viewports.set(v.id,v);
    });}
    getViewport(id:string){return this.viewports.get(id);}
    getViewports(){return [...this.viewports.values()];}
  }
  const lookup=(id:string)=>engines.flatMap(e=>e.getViewports()).find(v=>v.id===id);
  const toolGroup=()=>({addTool:vi.fn(),addViewport:vi.fn(),setToolActive:vi.fn(),setToolPassive:vi.fn()});
  const segmentation:any={
    addSegmentations:(ss:any[])=>ss.forEach(s=>states.set(s.segmentationId,{...s,representationData:{[s.representation.type]:s.representation.data}})),
    addRepresentationData:({segmentationId,type,data}:any)=>states.get(segmentationId).representationData[type]=data,
    addSegmentationRepresentations:(vid:string,rs:any[])=>reps.set(vid,[...(reps.get(vid)||[]),...rs]),
    addSurfaceRepresentationToViewport:(vid:string,rs:any[])=>reps.set(vid,rs.map(r=>({...r,type:'Surface'}))),
    removeSegmentationRepresentations:(vid:string)=>{reps.delete(vid);const v=lookup(vid);if(v)v.actors=v.actors.filter((a:any)=>!a.representationUID);},
    removeSegmentation:(id:string)=>states.delete(id),
    state:{addColorLUT:(lut:any)=>{const i=luts.size;luts.set(i,lut);return i;},removeColorLUT:(i:number)=>luts.delete(i),
      getSegmentationRepresentation:(vid:string,s:any)=>reps.get(vid)?.find(r=>r.segmentationId===s.segmentationId&&r.type===s.type)},
    config:{style:{setStyle:vi.fn()},visibility:{setSegmentationRepresentationVisibility:vi.fn(),setSegmentIndexVisibility:vi.fn()}},
  };
  return {Node,volumes,geometries,states,reps,luts,engines,actor,Engine,lookup,toolGroup,segmentation,
    failNextGeometry:()=>failGeometry=true, createGeometry:(id:string,options:any)=>{if(failGeometry){failGeometry=false;throw Error("synthetic conversion failure");}geometries.set(id,options.geometryData);}};
});
vi.mock('@cornerstonejs/tools/segmentation/SegmentationRepresentationDisplayRegistry',()=>({registerSegmentationRepresentationDisplay:vi.fn()}));
vi.mock('@cornerstonejs/core',()=>({
  Enums:{OrientationAxis:{AXIAL:'z',CORONAL:'y',SAGITTAL:'x'},ViewportType:{ORTHOGRAPHIC:'MPR',VOLUME_3D:'3D'},GeometryType:{SURFACE:'Surface'},Events:{IMAGE_RENDERED:'rendered'}},
  RenderingEngine:f.Engine,init:vi.fn(),metaData:{get:vi.fn(),addProvider:vi.fn(),removeProvider:vi.fn()},
  cache:{removeVolumeLoadObject:(id:string)=>f.volumes.delete(id),removeImageLoadObject:vi.fn(),removeGeometryLoadObject:(id:string)=>f.geometries.delete(id)},
  volumeLoader:{createLocalVolume:(id:string,options:any)=>{f.volumes.set(id,options);return {imageIds:[]};},
    createLocalLabelmapVolume:(options:any,id:string)=>{f.volumes.set(id,options);return {imageIds:[]};}},
  geometryLoader:{createAndCacheGeometry:f.createGeometry},
}));
vi.mock('@cornerstonejs/tools',()=>({
  ...Object.fromEntries(['CrosshairsTool','PanTool','StackScrollTool','TrackballRotateTool','WindowLevelTool','ZoomTool'].map(n=>[n,{toolName:n}])),
  Enums:{MouseBindings:{Primary:1,Auxiliary:2,Secondary:3,Wheel:4},SegmentationRepresentations:{Labelmap:'Labelmap',Surface:'Surface'}},
  ToolGroupManager:{createToolGroup:f.toolGroup,getToolGroup:f.toolGroup,destroyToolGroup:vi.fn()},
  init:vi.fn(),addTool:vi.fn(),segmentation:f.segmentation,
  synchronizers:{createZoomPanSynchronizer:()=>({add:vi.fn(),destroy:vi.fn()}),createVOISynchronizer:()=>({add:vi.fn(),destroy:vi.fn()})},
}));
vi.mock('@cornerstonejs/tools/tools/displayTools/Labelmap/labelmapDisplay',()=>({default:{render:async(v:any,r:any)=>{
  v.actors.push({uid:'labelmap',representationUID:r.segmentationId+'-Labelmap',actor:f.actor('vtkVolume')});
}}}));
vi.mock('@cornerstonejs/tools/tools/displayTools/Surface/surfaceDisplay',()=>({default:{render:async(v:any,r:any)=>{
  f.states.get(r.segmentationId).representationData.Surface.geometryIds.forEach((id:string,label:number)=>
    v.actors.push({uid:id,representationUID:r.segmentationId+'-Surface-'+label,actor:f.actor('vtkActor',f.geometries.get(id))}));
}}}));
vi.mock('@cornerstonejs/tools/segmentation/helpers/getSegmentationActor',()=>({getSurfaceActorEntry:(vid:string,id:string,label:number)=>
  f.lookup(vid)?.actors.find((a:any)=>a.representationUID===id+'-Surface-'+label)}));
import { mountViewer } from './viewer';
const nodes=(root:any):any[]=>[root,...root.children.flatMap(nodes)];
const click=(root:any,text:string)=>nodes(root).find(n=>n.tag==='button'&&n.textContent===text).dispatchEvent(new Event('click'));
const input=(root:any,label:string)=>nodes(root).find(n=>n.tag==='label'&&(n.textContent===label||n.children.some((c:any)=>c.textContent.trim()===label))).children.find((c:any)=>c.tag==='input'||c.tag==='select');
const envelope=(value=80,empty=false,registration=false):ParsedEnvelope=>{
  const data=new Uint16Array(8*9*10);if(!empty){data[122]=2;data[563]=513;}
  const image={name:'image' as const,role:'scalar' as const,dtype:'uint8' as const,shapeZYX:[8,9,10] as [number,number,number],data:new Uint8Array(data.length).fill(value)};
  return {schema:'unused',viewerMode:registration?'registration':'segmentation',coordinateMode:'index-space',spacingSource:'protocol',spacingZYX:[2,1,1],originXYZ:[3,4,5],directionXYZ:[1,0,0,0,1,0,0,0,1],segments:[],context:{},
    volumes:registration?['fixed','moving','registered'].map(name=>({...image,name})):[image,{name:'prediction',role:'labelmap',dtype:'uint16',shapeZYX:image.shapeZYX,data}]} as ParsedEnvelope;
};
async function mount(e= envelope(),cancelled=()=>false){
  const parent=new f.Node();const bag=new CleanupBag();const state:any={};
  await mountViewer({parentElement:parent,key:'test',setStateValue:(k:string,v:any)=>state[k]=v} as any,e,bag,cancelled);
  return {parent,bag,state,engine:f.engines.at(-1),v3:f.engines.at(-1).getViewports().find((v:any)=>v.type==='3D')};
}
beforeEach(()=>{
  f.engines.length=0;[f.volumes,f.geometries,f.states,f.reps,f.luts].forEach(x=>x.clear());
  vi.stubGlobal('document',{createElement:(tag:string)=>new f.Node(tag),createTextNode:(text:string)=>Object.assign(new f.Node('text'),{textContent:text})});
  vi.stubGlobal('ResizeObserver',class {observe(){} disconnect(){}});
  vi.stubGlobal('IntersectionObserver',class {observe(){} disconnect(){}});
  vi.stubGlobal('requestAnimationFrame',()=>1);vi.stubGlobal('cancelAnimationFrame',vi.fn());
});
describe('segmentation mount/control/cleanup routing',()=>{
  it('routes image only to MPR, real prediction surfaces to 3D; image independence',async()=>{
    const a=await mount(envelope(0));const b=await mount(envelope(255));
    const mesh=(m:any)=>m.v3.actors.map((a:any)=>({points:a.actor.geometry.points,polys:a.actor.geometry.polys,material:a.actor.property}));
    expect(mesh(a)).toEqual(mesh(b));expect(a.v3.actors).toHaveLength(2);
    for(const m of [a,b]){
      expect(m.v3.setVolumes).not.toHaveBeenCalled();
      expect(m.v3.actors.every((a:any)=>a.actor.kind==='vtkActor')).toBe(true);
      expect(m.engine.getViewports().filter((v:any)=>v.type==='MPR').every((v:any)=>v.actors.length===2)).toBe(true);
      expect(m.state.viewer_ready).toBe(true);
    }
    expect(a.engine.getViewports()[0].actors[0].input.scalarData).not.toEqual(b.engine.getViewports()[0].actors[0].input.scalarData);
    a.bag.close();b.bag.close();expect(f.geometries.size+f.volumes.size+f.states.size+f.reps.size+f.luts.size).toBe(0);
  });
  it('separates visibility, opacity and reset; synchronizes uint16 label selection',async()=>{
    const m=await mount();const before=m.v3.resetCamera.mock.calls.length;
    const image=input(m.parent,'切面原图');image.checked=false;image.dispatchEvent(new Event('change'));
    expect(m.v3.actors.every((a:any)=>a.actor.visible)).toBe(true);
    expect(m.engine.getViewports().filter((v:any)=>v.type==='MPR').every((v:any)=>!v.actors[0].actor.visible)).toBe(true);
    const alpha=input(m.parent,'3D 不透明度 ');alpha.value='.3';alpha.dispatchEvent(new Event('input'));
    expect(m.v3.actors.every((a:any)=>a.actor.property.opacity===.3)).toBe(true);
    expect(m.v3.resetCamera.mock.calls.length).toBe(before);
    const select=input(m.parent,'标签 ');select.value='513';select.dispatchEvent(new Event('change'));
    expect(m.v3.actors.map((a:any)=>a.actor.visible)).toEqual([false,true]);
    expect(f.segmentation.config.visibility.setSegmentIndexVisibility).toHaveBeenCalledWith(m.v3.id,expect.objectContaining({type:'Surface'}),513,true);
    const overlay=input(m.parent,'切面叠加');overlay.checked=false;overlay.dispatchEvent(new Event('change'));
    select.value='2';select.dispatchEvent(new Event('change'));
    for(const v of m.engine.getViewports().filter((v:any)=>v.type==='MPR')) expect(f.segmentation.config.visibility.setSegmentIndexVisibility).toHaveBeenCalledWith(v.id,expect.objectContaining({type:'Labelmap'}),2,false);
    overlay.checked=true;overlay.dispatchEvent(new Event('change'));
    select.value='513';select.dispatchEvent(new Event('change'));
    click(m.parent,'Reset');expect(m.v3.resetProperties).not.toHaveBeenCalled();expect(m.v3.setVolumes).not.toHaveBeenCalled();
    expect(m.v3.actors.map((a:any)=>a.actor.visible)).toEqual([false,true]);m.bag.close();
  });
  it('handles empty foreground and cancellation without source fallback or resource leaks',async()=>{
    const m=await mount(envelope(80,true));expect(m.v3.actors).toEqual([]);click(m.parent,'Reset');
    expect(nodes(m.parent).some(n=>n.textContent==='当前预览没有可显示的预测前景')).toBe(true);m.bag.close();
    let checks=0;const cancelled=await mount(envelope(),()=>++checks>2);cancelled.bag.close();
    expect(f.geometries.size+f.volumes.size+f.states.size+f.reps.size+f.luts.size).toBe(0);
    for(let i=0;i<3;i++){const x=await mount();x.bag.close();}
    expect(f.geometries.size+f.volumes.size+f.states.size+f.reps.size+f.luts.size).toBe(0);
  });
  it('keeps uint16 values exact in the GPU display copy and preserves MPR on Surface failure',async()=>{
    f.failNextGeometry();
    const e=envelope();const original=e.volumes[1]!.data.slice();
    const m=await mount(e);
    expect(m.state.viewer_ready).toBe(false);expect(m.state.viewer_error_code).toBe('PREDICTION_SURFACE_FAILED');
    expect(m.v3.actors).toHaveLength(0);
    expect(m.engine.getViewports().filter((v:any)=>v.type==='MPR').every((v:any)=>v.actors.length===2)).toBe(true);
    const display=[...f.volumes.values()].find(v=>v.scalarData instanceof Float32Array);
    expect(Array.from(display.scalarData)).toEqual(Array.from(original));
    expect(e.volumes[1]!.data).toEqual(original);
    click(m.parent,'Reset');expect(m.v3.setVolumes).not.toHaveBeenCalled();m.bag.close();
    expect(f.geometries.size+f.volumes.size+f.states.size+f.reps.size+f.luts.size).toBe(0);
  });
  it('retains registration scalar volume and preset/reset path',async()=>{
    const m=await mount(envelope(80,false,true));expect(m.v3.setVolumes).toHaveBeenCalled();
    expect(m.v3.actors[0].actor.kind).toBe('vtkVolume');expect(m.v3.setProperties).toHaveBeenCalledWith(expect.objectContaining({preset:'MR-Default'}),expect.any(String));
    click(m.parent,'Reset');expect(m.v3.resetProperties).toHaveBeenCalled();m.bag.close();
  });
});
