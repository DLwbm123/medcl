import { describe, it, expect } from "vitest";
import { predictionSurface, presentLabels, labelColor } from "./predictionSurface";
import type { VolumeBlock } from "./volumeProtocol";

const block = (shape: [number, number, number], data = new Uint16Array(shape.reduce((a, b) => a * b))): VolumeBlock =>
  ({ name: "prediction", role: "labelmap", dtype: "uint16", shapeZYX: shape, data });
const bounds = (points: number[]) => [0, 1, 2].map((axis) => {
  const values = points.filter((_, i) => i % 3 === axis);
  return [Math.min(...values), Math.max(...values)];
});

describe("per-label prediction surfaces", () => {
  it("preserves separated, noncontinuous uint16 labels and world geometry", () => {
    const p = block([8, 9, 10]);
    for (let z=1; z<4; z++) for (let y=2; y<5; y++) for (let x=1; x<3; x++) p.data[(z*9+y)*10+x]=2;
    for (let z=5; z<7; z++) for (let y=6; y<8; y++) for (let x=7; x<9; x++) p.data[(z*9+y)*10+x]=513;
    const before = p.data.slice();
    expect(presentLabels(p)).toEqual([2,513]);
    const a=predictionSurface(p,2,[3,2,1],[10,20,30]);
    const b=predictionSurface(p,513,[3,2,1],[10,20,30]);
    expect(bounds(a.points)).toEqual([[10.5,12.5],[23,29],[31.5,40.5]]);
    expect(bounds(b.points)).toEqual([[16.5,18.5],[31,35],[43.5,49.5]]);
    expect(a.polys.length).toBeGreaterThan(12);
    expect(b.polys.length).toBeGreaterThan(12);
    expect(p.data).toEqual(before);
    expect(labelColor(513)).toEqual(labelColor(513));
    expect(labelColor(2)).not.toEqual(labelColor(7));
  });
  it("renders seven foregrounds with distinct stable colors and independent meshes", () => {
    const p = block([4, 6, 30]);
    for (let label = 1; label <= 7; label++)
      for (let z = 1; z <= 2; z++) for (let y = 2; y <= 3; y++) p.data[(z * 6 + y) * 30 + label * 4] = label;
    expect(presentLabels(p)).toEqual([1,2,3,4,5,6,7]);
    expect(new Set(presentLabels(p).map(l => labelColor(l).join(","))).size).toBe(7);
    for (const label of presentLabels(p)) {
      const mesh = predictionSurface(p, label, [1,2,2], [0,0,0]);
      expect(bounds(mesh.points)[0]).toEqual([label * 8 - 1, label * 8 + 1]);
      expect(mesh.polys.length).toBeGreaterThan(12);
    }
  });
  it("has no image input: image changes cannot change geometry or colors", () => {
    const p=block([4,5,6]); p.data[42]=7; p.data[70]=2;
    const render = (image: Uint8Array) => ({image, meshes: presentLabels(p).map(l => ({...predictionSurface(p,l,[2,1,1],[0,0,0]),color:labelColor(l)}))});
    const a=render(new Uint8Array(120)); const b=render(new Uint8Array(120).fill(255));
    expect(a.meshes).toEqual(b.meshes); expect(a.image).not.toEqual(b.image);
  });
  it("closes thin and boundary-touching/full-volume labels without discarding voxels", () => {
    for (const shape of [[1,1,1],[1,3,4],[2,3,4]] as [number,number,number][]) {
      const p=block(shape); p.data.fill(7);
      const mesh=predictionSurface(p,7,[2,3,4],[0,0,0]);
      expect(bounds(mesh.points)).toEqual([[-2,(shape[2]-.5)*4],[-1.5,(shape[1]-.5)*3],[-1,(shape[0]-.5)*2]]);
      // Every triangle edge is shared twice: closed 3D surface even for Z=1.
      const edges=new Map<string,number>();
      for(let i=0;i<mesh.polys.length;) {
        const n=mesh.polys[i++]!; const ids=mesh.polys.slice(i,i+n); i+=n;
        ids.forEach((id,j)=>{const key=[id,ids[(j+1)%n]!].sort((a,b)=>a-b).join(',');edges.set(key,(edges.get(key)||0)+1);});
      }
      expect([...edges.values()].every(n=>n===2)).toBe(true);
    }
  });
  it("treats background and missing foreground as valid empty geometry", () => {
    const p=block([1,2,3]); expect(presentLabels(p)).toEqual([]);
    expect(predictionSurface(p,0,[1,1,1],[0,0,0])).toEqual({points:[],polys:[]});
    expect(predictionSurface(p,7,[1,1,1],[0,0,0])).toEqual({points:[],polys:[]});
  });
});
