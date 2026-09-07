import vtkImageData from "@kitware/vtk.js/Common/DataModel/ImageData";
import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray";
import type vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData";
// vtk.js 36.4.1 ships this filter's implementation, but no declaration.
// @ts-expect-error Missing upstream declaration, pinned implementation checked locally.
import vtkImageMarchingCubes from "@kitware/vtk.js/Filters/General/ImageMarchingCubes";
import type { VolumeBlock } from "./volumeProtocol";

export const presentLabels = (prediction: VolumeBlock): number[] =>
  [...new Set(prediction.data)].filter((label) => label !== 0).sort((a, b) => a - b);

export const labelColor = (label: number): [number, number, number, number] => {
  const palette = [[38, 200, 122], [255, 181, 71], [96, 165, 250], [207, 122, 232], [255, 112, 137], [67, 217, 214], [232, 222, 85]];
  return [...palette[(label - 1) % palette.length]!, 255] as [number, number, number, number];
};

/** Binary, per-label conversion. Never receives the scalar image. World XYZ output.
 * One background-voxel halo closes boundary-touching objects and single-slice data.
 * No smoothing, component filtering, label remapping, or voxel edits.
 */
export function predictionSurface(prediction: VolumeBlock, label: number,
  spacingZYX: [number, number, number], originXYZ: [number, number, number]): { points: number[]; polys: number[] } {
  const [nz, ny, nx] = prediction.shapeZYX;
  const lo = [nx, ny, nz], hi = [-1, -1, -1];
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    if (prediction.data[(z * ny + y) * nx + x] !== label) continue;
    [x, y, z].forEach((v, axis) => { lo[axis] = Math.min(lo[axis]!, v); hi[axis] = Math.max(hi[axis]!, v); });
  }
  if (label === 0 || hi[0]! < 0) return { points: [], polys: [] };
  const dims = lo.map((v, axis) => hi[axis]! - v + 3) as [number, number, number];
  const binary = new Uint8Array(dims[0] * dims[1] * dims[2]);
  for (let z = lo[2]!; z <= hi[2]!; z++) for (let y = lo[1]!; y <= hi[1]!; y++) for (let x = lo[0]!; x <= hi[0]!; x++) {
    binary[((z - lo[2]! + 1) * dims[1] + y - lo[1]! + 1) * dims[0] + x - lo[0]! + 1] =
      Number(prediction.data[(z * ny + y) * nx + x] === label);
  }
  const spacing = [...spacingZYX].reverse() as [number, number, number];
  const image = vtkImageData.newInstance();
  image.setDimensions(...dims);
  image.setSpacing(spacing);
  image.setOrigin(originXYZ.map((v, i) => v + (lo[i]! - 1) * spacing[i]!) as [number, number, number]);
  const scalars = vtkDataArray.newInstance({ name: "prediction-binary", values: binary, numberOfComponents: 1 });
  image.getPointData().setScalars(scalars);
  const filter = vtkImageMarchingCubes.newInstance({ contourValue: 0.5, computeNormals: false, mergePoints: true }) as {
    setInputData(data: typeof image): void; getOutputData(): vtkPolyData; delete(): void;
  };
  let output: vtkPolyData | undefined;
  try {
    filter.setInputData(image);
    output = filter.getOutputData();
    const points = Array.from(output.getPoints().getData());
    const polys = Array.from(output.getPolys().getData());
    if (!points.length || !polys.length || !points.every(Number.isFinite)) throw new Error("prediction_surface_conversion_failed");
    return { points, polys };
  } finally {
    output?.delete(); filter.delete(); scalars.delete(); image.delete();
  }
}
