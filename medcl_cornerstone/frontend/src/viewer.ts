import {
  Enums as CoreEnums,
  RenderingEngine,
  Types,
  cache,
  init as initCore,
  volumeLoader,
  geometryLoader,
  metaData,
} from "@cornerstonejs/core";
import {
  CrosshairsTool,
  Enums as ToolEnums,
  PanTool,
  StackScrollTool,
  ToolGroupManager,
  TrackballRotateTool,
  WindowLevelTool,
  ZoomTool,
  addTool,
  init as initTools,
  segmentation,
  synchronizers,
} from "@cornerstonejs/tools";
import type { FrontendRendererArgs } from "@streamlit/component-v2-lib";
import surfaceDisplay from "@cornerstonejs/tools/tools/displayTools/Surface/surfaceDisplay";
import labelmapDisplay from "@cornerstonejs/tools/tools/displayTools/Labelmap/labelmapDisplay";
import { getSurfaceActorEntry } from "@cornerstonejs/tools/segmentation/helpers/getSegmentationActor";
import { registerSegmentationRepresentationDisplay } from "@cornerstonejs/tools/segmentation/SegmentationRepresentationDisplayRegistry";
import type vtkActor from "@kitware/vtk.js/Rendering/Core/Actor";
import { presentLabels, predictionSurface, labelColor } from "./predictionSurface";
import { CleanupBag } from "./lifecycle";
import viewerStyles from "./styles.css?inline";
import { mprBindingPlan, MprPrimaryTool, MprTool, MouseBinding } from "./interaction";
import {
  defaultRegistrationLayer,
  LayerMode,
  ParsedEnvelope,
  registrationLayerChoices,
  stableId,
  VolumeBlock,
  VolumeName,
  xyzDimensions,
  xyzSpacing,
} from "./volumeProtocol";

export interface ViewerState extends Record<string, unknown> {
  viewer_ready: boolean;
  viewer_error_code: string | null;
  selected_segment: number | null;
  layer_mode: string | null;
}

type Args = FrontendRendererArgs<ViewerState, Uint8Array | ArrayBuffer>;
type VolumeViewport = Types.IVolumeViewport & {
  getSliceIndex(): number;
  getNumberOfSlices(): number;
  resetCamera(): boolean;
};

const viewportSpecs = [
  ["axial", "Z / axial-like", CoreEnums.OrientationAxis.AXIAL],
  ["coronal", "Y / coronal-like", CoreEnums.OrientationAxis.CORONAL],
  ["sagittal", "X / sagittal-like", CoreEnums.OrientationAxis.SAGITTAL],
] as const;
const toolClasses = [WindowLevelTool, PanTool, ZoomTool, StackScrollTool, CrosshairsTool, TrackballRotateTool];
const normalizedVoi = { lower: 0, upper: 255 } as const;
let initialized: Promise<void> | undefined;

const mprToolNames: Record<MprTool, string> = {
  windowLevel: WindowLevelTool.toolName,
  pan: PanTool.toolName,
  zoom: ZoomTool.toolName,
  crosshairs: CrosshairsTool.toolName,
  stackScroll: StackScrollTool.toolName,
};
const mouseBindings: Record<MouseBinding, ToolEnums.MouseBindings> = {
  Primary: ToolEnums.MouseBindings.Primary,
  Auxiliary: ToolEnums.MouseBindings.Auxiliary,
  Secondary: ToolEnums.MouseBindings.Secondary,
  Wheel: ToolEnums.MouseBindings.Wheel,
};

const immutableSurfaceViewports = new Set<string>();

const initialize = (): Promise<void> => {
  if (!initialized) {
    initialized = Promise.resolve().then(() => {
      initCore();
      initTools();
      // 5.8.2's default Surface update callback assumes PolySeg is installed.
      // These preview meshes are immutable and rebuilt on each envelope mount.
      registerSegmentationRepresentationDisplay(ToolEnums.SegmentationRepresentations.Surface, {
        ...surfaceDisplay,
        getUpdateFunction: (viewport) => immutableSurfaceViewports.has(viewport.id) ? undefined : surfaceDisplay.getUpdateFunction(viewport),
      });
      for (const tool of toolClasses) addTool(tool);
    });
  }
  return initialized;
};

const element = <K extends keyof HTMLElementTagNameMap>(tag: K, className?: string, text?: string): HTMLElementTagNameMap[K] => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};

const button = (label: string, title: string): HTMLButtonElement => {
  const node = element("button", "medcl-button", label);
  node.type = "button";
  node.title = title;
  return node;
};

const metadata = (
  dimensions: [number, number, number],
  spacing: [number, number, number],
  frame: string,
  dtype: VolumeBlock["dtype"],
): Types.Metadata => {
  const bits = dtype === "uint8" ? 8 : dtype === "uint16" ? 16 : 32;
  return {
    BitsAllocated: bits,
    BitsStored: bits,
    SamplesPerPixel: 1,
    HighBit: bits - 1,
    PhotometricInterpretation: "MONOCHROME2",
    PixelRepresentation: 0,
    Modality: "OT",
    ImageOrientationPatient: [1, 0, 0, 0, 1, 0],
    PixelSpacing: [spacing[1], spacing[0]],
    FrameOfReferenceUID: frame,
    Columns: dimensions[0],
    Rows: dimensions[1],
    voiLut: [{ windowWidth: 256, windowCenter: 127.5 }],
    VOILUTFunction: "LINEAR",
  };
};

const createVolume = (block: VolumeBlock, id: string, envelope: ParsedEnvelope, frame: string): string[] => {
  const dimensions = xyzDimensions(block.shapeZYX);
  const spacing = xyzSpacing(envelope.spacingZYX);
  // Exact numeric display copy: all uint16 labels are representable in float32.
  // vtk.js's normalized unsigned-short texture path loses categorical overlays.
  const floatLabels = envelope.viewerMode === "segmentation" && block.role === "labelmap" && block.dtype === "uint16";
  const options = {
    metadata: metadata(dimensions, spacing, frame, floatLabels ? "float32" : block.dtype),
    dimensions,
    spacing,
    origin: envelope.originXYZ,
    direction: envelope.directionXYZ,
    scalarData: floatLabels ? Float32Array.from(block.data) : block.data as Types.PixelDataTypedArray,
  };
  const volume = block.role === "labelmap"
    ? volumeLoader.createLocalLabelmapVolume(options, id)
    : volumeLoader.createLocalVolume(id, options);
  return [...volume.imageIds];
};

const layerInputs = (mode: LayerMode, ids: Map<VolumeName, string>): Types.IVolumeInput[] => {
  const names: VolumeName[] = mode === "fixed-moving" ? ["fixed", "moving"]
    : mode === "fixed-registered" ? ["fixed", "registered"] : [mode];
  return names.map((name) => ({ volumeId: ids.get(name)! }));
};

const configureToolGroups = (engineId: string, mprIds: string[], volume3dId: string, unique: string): string[] => {
  const mprGroupId = `${unique}-mpr`;
  const volumeGroupId = `${unique}-3d`;
  const mpr = ToolGroupManager.createToolGroup(mprGroupId);
  const volume3d = ToolGroupManager.createToolGroup(volumeGroupId);
  if (!mpr || !volume3d) throw new Error("tool_group_failed");
  for (const name of [WindowLevelTool.toolName, PanTool.toolName, ZoomTool.toolName, StackScrollTool.toolName, CrosshairsTool.toolName]) mpr.addTool(name);
  for (const name of [TrackballRotateTool.toolName, PanTool.toolName, ZoomTool.toolName]) volume3d.addTool(name);
  for (const id of mprIds) mpr.addViewport(id, engineId);
  volume3d.addViewport(volume3dId, engineId);
  applyMprTool(mpr, "windowLevel");
  volume3d.setToolActive(TrackballRotateTool.toolName, { bindings: [{ mouseButton: ToolEnums.MouseBindings.Primary }] });
  volume3d.setToolActive(PanTool.toolName, { bindings: [{ mouseButton: ToolEnums.MouseBindings.Auxiliary }] });
  volume3d.setToolActive(ZoomTool.toolName, { bindings: [
    { mouseButton: ToolEnums.MouseBindings.Secondary },
    { mouseButton: ToolEnums.MouseBindings.Wheel },
  ] });
  return [mprGroupId, volumeGroupId];
};

const applyMprTool = (group: NonNullable<ReturnType<typeof ToolGroupManager.getToolGroup>>, primary: MprPrimaryTool): void => {
  const plan = mprBindingPlan(primary);
  for (const name of Object.values(mprToolNames)) group.setToolPassive(name, { removeAllBindings: true });
  for (const [tool, bindings] of Object.entries(plan) as [MprTool, MouseBinding[]][]) {
    if (bindings.length) group.setToolActive(mprToolNames[tool], {
      bindings: bindings.map((mouseButton) => ({ mouseButton: mouseBindings[mouseButton] })),
    });
  }
};

function makeShell(parent: HTMLElement | ShadowRoot, envelope: ParsedEnvelope): {
  toolbar: HTMLDivElement;
  grid: HTMLDivElement;
  status: HTMLDivElement;
  panes: Map<string, HTMLDivElement>;
  sliceLabels: Map<string, HTMLSpanElement>;
} {
  parent.replaceChildren();
  const root = element("section", "medcl-viewer");
  root.setAttribute("aria-label", "MedCL medical volume viewer");
  const heading = element("div", "medcl-heading");
  heading.append(element("strong", "", envelope.viewerMode === "segmentation" ? "Segmentation volume" : "Registration volume"));
  const shape = envelope.volumes[0]!.shapeZYX;
  const context = envelope.context;
  const score = context.score === undefined || context.score === null
    ? undefined
    : `${envelope.viewerMode === "segmentation" ? "Dice" : "TRE"} ${context.score.toFixed(4)}${envelope.viewerMode === "registration" ? " mm" : ""}`;
  const details = [
    context.stage === undefined ? undefined : `stage ${context.stage}`,
    context.task_id === undefined ? undefined : `task ${context.task_id}`,
    context.case_id === undefined ? undefined : `case ${context.case_id}`,
    score,
    `ZYX ${shape.join("×")}`,
    `spacing ${envelope.spacingZYX.map((value) => Number(value.toPrecision(6))).join("/")} (${envelope.spacingSource === "protocol" ? "protocol" : "index-space default"})`,
    envelope.coordinateMode === "index-space"
      ? "array index space; patient orientation unverified"
      : "fixed display grid; patient orientation unverified",
    context.downsampled ? "downsampled preview" : "native preview grid",
  ].filter((value): value is string => value !== undefined);
  heading.append(element("span", "medcl-details", details.join(" · ")));
  const toolbar = element("div", "medcl-toolbar");
  toolbar.setAttribute("role", "toolbar");
  const grid = element("div", "medcl-grid");
  const panes = new Map<string, HTMLDivElement>();
  const sliceLabels = new Map<string, HTMLSpanElement>();
  for (const [id, label] of [...viewportSpecs.map(([id, label]) => [id, label] as const), ["volume3d", envelope.viewerMode === "segmentation" ? "预测分割 · 3D" : "3D volume"] as const]) {
    const card = element("div", "medcl-card");
    const title = element("div", "medcl-card-title");
    title.append(element("span", "", label));
    const slice = element("span", "medcl-slice", id === "volume3d" ? "rotate: left drag" : "slice —");
    title.append(slice);
    const pane = element("div", "medcl-pane");
    pane.setAttribute("aria-label", label);
    card.append(title, pane);
    grid.append(card);
    panes.set(id, pane);
    sliceLabels.set(id, slice);
  }
  const status = element("div", "medcl-status", "Initializing viewer…");
  status.setAttribute("role", "status");
  root.append(heading, toolbar, grid, status);
  const stylesheet = element("style");
  stylesheet.textContent = viewerStyles;
  parent.append(stylesheet, root);
  return { toolbar, grid, status, panes, sliceLabels };
}

const addToolButtons = (toolbar: HTMLDivElement, mprGroupId: string, engine: RenderingEngine, onReset?: () => void): void => {
  const actions: Array<[string, string, MprPrimaryTool]> = [
    ["W/L", "Window / level", "windowLevel"],
    ["Pan", "Pan all MPR views", "pan"],
    ["Zoom", "Zoom all MPR views", "zoom"],
    ["Crosshair", "Linked crosshair navigation", "crosshairs"],
  ];
  let active: MprPrimaryTool = "windowLevel";
  const controls = new Map<MprPrimaryTool, HTMLButtonElement>();
  for (const [label, title, tool] of actions) {
    const control = button(label, title);
    controls.set(tool, control);
    control.setAttribute("aria-pressed", String(tool === active));
    control.classList.toggle("medcl-button-active", tool === active);
    control.addEventListener("click", () => {
      if (tool === active) return;
      const group = ToolGroupManager.getToolGroup(mprGroupId);
      if (!group) return;
      applyMprTool(group, tool);
      active = tool;
      for (const [name, item] of controls) {
        item.setAttribute("aria-pressed", String(name === active));
        item.classList.toggle("medcl-button-active", name === active);
      }
    });
    toolbar.append(control);
  }
  const reset = button("Reset", "Reset cameras and display properties");
  reset.addEventListener("click", () => {
    if (onReset) { onReset(); return; }
    for (const viewport of engine.getViewports()) {
      const volume = viewport as VolumeViewport;
      volume.resetProperties?.();
      volume.resetCamera?.();
    }
    engine.render();
  });
  toolbar.append(reset);
}

const addOverlayControls = (
  toolbar: HTMLDivElement,
  viewportIds: string[],
  segmentationId: string,
  segments: number[],
  args: Args,
  onSelect3D?: (selected: number | null) => void,
): void => {
  let selectedLabel: number | null = null;
  const visibleLabel = element("label", "medcl-control");
  const visible = element("input") as HTMLInputElement;
  visible.type = "checkbox";
  visible.checked = true;
  visibleLabel.append(visible, document.createTextNode(onSelect3D ? " 切面叠加" : " overlay"));
  visible.addEventListener("change", () => {
    for (const viewportId of viewportIds) {
      segmentation.config.visibility.setSegmentationRepresentationVisibility(
        viewportId,
        { segmentationId, type: ToolEnums.SegmentationRepresentations.Labelmap },
        visible.checked,
      );
      if (onSelect3D) for (const segment of segments) {
        segmentation.config.visibility.setSegmentIndexVisibility(viewportId,
          { segmentationId, type: ToolEnums.SegmentationRepresentations.Labelmap }, segment,
          visible.checked && (selectedLabel === null || selectedLabel === segment));
      }
    }
  });
  const opacityLabel = element("label", "medcl-control", onSelect3D ? "切面透明度 " : "Opacity ");
  const opacity = element("input") as HTMLInputElement;
  opacity.type = "range";
  opacity.min = "0";
  opacity.max = "1";
  opacity.step = "0.05";
  opacity.value = "0.45";
  opacity.addEventListener("input", () => {
    segmentation.config.style.setStyle(
      { segmentationId, type: ToolEnums.SegmentationRepresentations.Labelmap },
      { fillAlpha: Number(opacity.value), fillAlphaInactive: Number(opacity.value), renderOutline: !onSelect3D, renderOutlineInactive: !onSelect3D },
      true,
    );
  });
  opacityLabel.append(opacity);
  toolbar.append(visibleLabel, opacityLabel);
  if (segments.length) {
    const segmentLabel = element("label", "medcl-control", onSelect3D ? "标签 " : "Segment ");
    const select = element("select") as HTMLSelectElement;
    const all = element("option", "", onSelect3D ? "全部标签" : "All");
    all.value = "all";
    select.append(all);
    for (const segment of segments) {
      const option = element("option", "", `Label ${segment}`);
      option.value = String(segment);
      select.append(option);
    }
    select.addEventListener("change", () => {
      const selected = selectedLabel = select.value === "all" ? null : Number(select.value);
      for (const viewportId of viewportIds) {
        for (const segment of segments) {
          segmentation.config.visibility.setSegmentIndexVisibility(
            viewportId,
            { segmentationId, type: ToolEnums.SegmentationRepresentations.Labelmap },
            segment,
            (!onSelect3D || visible.checked) && (selected === null || selected === segment),
          );
        }
      }
      onSelect3D?.(selected);
      args.setStateValue("selected_segment", selected);
    });
    segmentLabel.append(select);
    toolbar.append(segmentLabel);
  }
};

const setRegistrationVolumes = async (
  mode: LayerMode,
  opacity: number,
  engine: RenderingEngine,
  viewportIds: string[],
  volume3dId: string,
  ids: Map<VolumeName, string>,
): Promise<void> => {
  const inputs = layerInputs(mode, ids);
  for (const viewportId of viewportIds) {
    const viewport = engine.getViewport<VolumeViewport>(viewportId);
    await viewport.setVolumes(inputs, false);
    for (const input of inputs) viewport.setProperties({ voiRange: normalizedVoi }, input.volumeId);
    if (inputs.length === 2) {
      viewport.setProperties({ voiRange: normalizedVoi, colormap: { opacity } }, inputs[1]!.volumeId);
    }
    viewport.resetCamera();
  }
  const volume3d = engine.getViewport<VolumeViewport>(volume3dId);
  await volume3d.setVolumes([inputs.at(-1)!], false);
  volume3d.setProperties({ voiRange: normalizedVoi, preset: "MR-Default" }, inputs.at(-1)!.volumeId);
  volume3d.resetCamera();
  engine.render();
};

const setRegistrationOpacity = (
  mode: LayerMode,
  opacity: number,
  engine: RenderingEngine,
  viewportIds: string[],
  ids: Map<VolumeName, string>,
): void => {
  const inputs = layerInputs(mode, ids);
  if (inputs.length !== 2) return;
  for (const viewportId of viewportIds) {
    engine.getViewport<VolumeViewport>(viewportId).setProperties(
      { voiRange: normalizedVoi, colormap: { opacity } }, inputs[1]!.volumeId,
    );
  }
  engine.render();
};

const addRegistrationControls = (
  toolbar: HTMLDivElement,
  envelope: ParsedEnvelope,
  args: Args,
  applyMode: (mode: LayerMode, opacity: number) => Promise<void>,
  applyOpacity: (mode: LayerMode, opacity: number) => void,
): LayerMode => {
  const initial = defaultRegistrationLayer(envelope);
  const layerLabel = element("label", "medcl-control", "Layers ");
  const select = element("select") as HTMLSelectElement;
  for (const choice of registrationLayerChoices(envelope)) {
    const option = element("option", "", choice.label);
    option.value = choice.value;
    option.selected = choice.value === initial;
    select.append(option);
  }
  const opacityLabel = element("label", "medcl-control", "Fusion ");
  const opacity = element("input") as HTMLInputElement;
  opacity.type = "range";
  opacity.min = "0";
  opacity.max = "1";
  opacity.step = "0.05";
  opacity.value = "0.5";
  select.addEventListener("change", () => {
    const mode = select.value as LayerMode;
    args.setStateValue("layer_mode", mode);
    void applyMode(mode, Number(opacity.value)).catch(() => args.setStateValue("viewer_error_code", "LAYER_UPDATE_FAILED"));
  });
  opacity.addEventListener("input", () => applyOpacity(select.value as LayerMode, Number(opacity.value)));
  layerLabel.append(select);
  opacityLabel.append(opacity);
  toolbar.append(layerLabel, opacityLabel);
  return initial;
};

export async function mountViewer(args: Args, envelope: ParsedEnvelope, bag: CleanupBag, cancelled: () => boolean): Promise<void> {
  const shell = makeShell(args.parentElement, envelope);
  if (!document.createElement("canvas").getContext("webgl2")) throw new Error("webgl_unavailable");
  await initialize();
  if (cancelled()) return;

  const unique = stableId("medcl", `${args.key}-${Date.now()}-${Math.random()}`);
  const engineId = `${unique}-engine`;
  const frame = `${unique}-frame`;
  const ids = new Map<VolumeName, string>();
  const imageIds = new Map<string, string[]>();
  bag.add(() => {
    for (const [volumeId, slices] of imageIds) {
      try { cache.removeVolumeLoadObject(volumeId); } catch { /* already evicted */ }
      for (const imageId of slices) {
        try { cache.removeImageLoadObject(imageId, { force: true }); } catch { /* already evicted */ }
      }
    }
  });
  for (const block of envelope.volumes) {
    const volumeId = `${unique}-${block.name}`;
    ids.set(block.name, volumeId);
    imageIds.set(volumeId, createVolume(block, volumeId, envelope, frame));
  }
  if (envelope.viewerMode === "segmentation") {
    // createLocalVolume in 5.8.2 gives every slice the same position.
    // Supply the real preview slice positions for MPR image/labelmap matching.
    const planes = new Map<string, object>();
    for (const slices of imageIds.values()) slices.forEach((imageId, z) => {
      planes.set(imageId, { ...metaData.get("imagePlaneModule", imageId),
        frameOfReferenceUID: frame,
        imagePositionPatient: envelope.originXYZ.map((v, axis) => v + z * envelope.spacingZYX[0] * envelope.directionXYZ[axis + 6]!),
      });
    });
    const provider = (type: string, imageId: unknown) => type === "imagePlaneModule" && typeof imageId === "string" ? planes.get(imageId) : undefined;
    metaData.addProvider(provider, 1000);
    bag.add(() => { metaData.removeProvider(provider); planes.clear(); });
  }
  const engine = new RenderingEngine(engineId);
  bag.add(() => engine.destroy());
  const mprIds = viewportSpecs.map(([id]) => `${unique}-${id}`);
  const volume3dId = `${unique}-volume3d`;
  engine.setViewports([
    ...viewportSpecs.map(([id, , orientation], index) => ({
      viewportId: mprIds[index]!,
      element: shell.panes.get(id)!,
      type: CoreEnums.ViewportType.ORTHOGRAPHIC,
      defaultOptions: { orientation, background: [0.025, 0.035, 0.055] as Types.RGB },
    })),
    {
      viewportId: volume3dId,
      element: shell.panes.get("volume3d")!,
      type: CoreEnums.ViewportType.VOLUME_3D,
      defaultOptions: { background: [0.025, 0.035, 0.055] as Types.RGB },
    },
  ]);
  const groupIds = configureToolGroups(engineId, mprIds, volume3dId, unique);
  bag.add(() => { for (const groupId of groupIds) ToolGroupManager.destroyToolGroup(groupId); });
  let resetSegmentation3D: (() => void) | undefined;
  addToolButtons(shell.toolbar, groupIds[0]!, engine, envelope.viewerMode === "segmentation" ? () => {
    for (const viewportId of mprIds) {
      const viewport = engine.getViewport<VolumeViewport>(viewportId);
      viewport.resetProperties();
      viewport.setProperties({ voiRange: normalizedVoi }, ids.get("image")!);
      viewport.resetCamera();
    }
    resetSegmentation3D?.();
    engine.render();
  } : undefined);

  const syncs = [
    synchronizers.createZoomPanSynchronizer(`${unique}-zoom-pan`),
    synchronizers.createVOISynchronizer(`${unique}-voi`, { syncInvertState: true, syncColormap: false }),
  ];
  bag.add(() => { for (const sync of syncs) sync.destroy(); });
  for (const sync of syncs) for (const viewportId of mprIds) sync.add({ renderingEngineId: engineId, viewportId });

  let segmentationId: string | undefined;
  let surfaceError: string | undefined;
  if (envelope.viewerMode === "segmentation") {
    const scalarId = ids.get("image")!;
    const prediction = envelope.volumes.find((block) => block.name === "prediction")!;
    const labels = presentLabels(prediction);
    const viewport3D = engine.getViewport<VolumeViewport>(volume3dId);
    immutableSurfaceViewports.add(volume3dId);
    bag.add(() => immutableSurfaceViewports.delete(volume3dId));
    // Only these three viewports can ever receive the source scalar image.
    for (const viewportId of mprIds) {
      const viewport = engine.getViewport<VolumeViewport>(viewportId);
      await viewport.setVolumes([{ volumeId: scalarId, actorUID: scalarId }], false);
      if (cancelled()) return;
      viewport.setProperties({ voiRange: normalizedVoi }, scalarId);
      viewport.resetCamera();
    }
    const imageLabel = element("label", "medcl-control");
    const imageVisible = element("input") as HTMLInputElement;
    imageVisible.type = "checkbox";
    imageVisible.checked = true;
    imageLabel.append(imageVisible, document.createTextNode(" 切面原图"));
    imageVisible.addEventListener("change", () => {
      for (const viewportId of mprIds) {
        engine.getViewport<VolumeViewport>(viewportId).getActor(scalarId)?.actor.setVisibility(imageVisible.checked);
      }
      engine.render();
    });
    shell.toolbar.append(imageLabel);
    const id = segmentationId = `${unique}-prediction-segmentation`;
    const geometryIds = new Map<number, string>();
    const lut: Types.ColorLUT = Array.from({ length: Math.max(0, ...labels) + 1 }, (_, i) =>
      i === 0 ? [0, 0, 0, 0] : labelColor(i));
    const lutIndex = segmentation.state.addColorLUT(lut);
    const clearSurfaces = () => {
      const actors = labels.map((label) => getSurfaceActorEntry(volume3dId, id, label)?.actor as vtkActor | undefined).filter(Boolean);
      try { segmentation.removeSegmentationRepresentations(volume3dId, { segmentationId: id }, true); } catch { /* already gone */ }
      for (const actor of actors) {
        const mapper = actor!.getMapper();
        mapper?.getInputData()?.delete(); mapper?.delete(); actor!.delete();
      }
      for (const geometryId of geometryIds.values()) cache.removeGeometryLoadObject(geometryId);
      geometryIds.clear();
    };
    bag.add(() => {
      clearSurfaces();
      for (const viewportId of mprIds) {
        try { segmentation.removeSegmentationRepresentations(viewportId, { segmentationId: id }, true); } catch { /* already gone */ }
      }
      try { segmentation.removeSegmentation(id); } catch { /* already gone */ }
      segmentation.state.removeColorLUT(lutIndex);
    });
    segmentation.addSegmentations([{
      segmentationId: id,
      representation: { type: ToolEnums.SegmentationRepresentations.Labelmap, data: { volumeId: ids.get("prediction")! } },
      config: { segments: Object.fromEntries(labels.map((label) => [label, { segmentIndex: label }])) },
    }]);
    for (const viewportId of mprIds) {
      segmentation.addSegmentationRepresentations(viewportId, [{ segmentationId: id,
        type: ToolEnums.SegmentationRepresentations.Labelmap, config: { colorLUTOrIndex: lutIndex } }]);
    }
    segmentation.config.style.setStyle(
      { segmentationId: id, type: ToolEnums.SegmentationRepresentations.Labelmap },
      // 5.8.2's slice outline shader still draws hidden labels with width zero.
      // Filled overlays respect per-label alpha and visibility, including uint16.
      { fillAlpha: 0.45, fillAlphaInactive: 0.45, renderOutline: false, renderOutlineInactive: false }, true,
    );
    // Await the actual display operation rather than treating registration as Ready.
    for (const viewportId of mprIds) {
      await labelmapDisplay.render(engine.getViewport(viewportId),
        segmentation.state.getSegmentationRepresentation(viewportId, { segmentationId: id, type: ToolEnums.SegmentationRepresentations.Labelmap })! as Parameters<typeof labelmapDisplay.render>[1]);
      if (cancelled()) return;
    }
    try {
      for (const label of labels) {
        // Yield between labels so unmount/case changes can cancel before registration.
        await new Promise<void>((resolve) => setTimeout(resolve, 0));
        if (cancelled()) return;
        const mesh = predictionSurface(prediction, label, envelope.spacingZYX, envelope.originXYZ);
        const geometryId = `${unique}-surface-${label}`;
        geometryLoader.createAndCacheGeometry(geometryId, { type: CoreEnums.GeometryType.SURFACE,
          geometryData: { id: geometryId, ...mesh, segmentIndex: label, frameOfReferenceUID: frame } });
        geometryIds.set(label, geometryId);
      }
      if (labels.length) {
        segmentation.addRepresentationData({ segmentationId: id, type: ToolEnums.SegmentationRepresentations.Surface, data: { geometryIds } });
        segmentation.addSurfaceRepresentationToViewport(volume3dId, [{ segmentationId: id, config: { colorLUTOrIndex: lutIndex } }]);
        await surfaceDisplay.render(viewport3D, segmentation.state.getSegmentationRepresentation(volume3dId,
          { segmentationId: id, type: ToolEnums.SegmentationRepresentations.Surface })!);
        if (cancelled()) return;
        if (viewport3D.getActors().length !== labels.length || labels.some((label) => !getSurfaceActorEntry(volume3dId, id, label))) {
          throw new Error("prediction_surface_render_failed");
        }
      } else {
        const empty = element("div", "medcl-empty-prediction", "当前预览没有可显示的预测前景");
        empty.setAttribute("role", "status");
        shell.panes.get("volume3d")!.append(empty);
      }
    } catch (error) {
      if (cancelled()) return;
      clearSurfaces();
      surfaceError = "PREDICTION_SURFACE_FAILED";
      console.error("Prediction Surface rendering failed", error);
      const notice = element("div", "medcl-empty-prediction", "预测三维显示失败；仍可使用二维切面。");
      notice.setAttribute("role", "alert");
      shell.panes.get("volume3d")!.append(notice);
    }
    let selectedLabel: number | null = null;
    const opacity = element("input") as HTMLInputElement;
    opacity.type = "range"; opacity.min = "0"; opacity.max = "1"; opacity.step = "0.05"; opacity.value = "1";
    const applyMaterials = (): void => {
      for (const label of labels) {
        const actor = getSurfaceActorEntry(volume3dId, id, label)?.actor as vtkActor | undefined;
        if (!actor) continue;
        actor.setVisibility(selectedLabel === null || selectedLabel === label);
        const color = labelColor(label);
        actor.getProperty().setColor(color[0] / 255, color[1] / 255, color[2] / 255);
        actor.getProperty().setOpacity(Number(opacity.value));
        actor.getProperty().setAmbient(0.25);
        actor.getProperty().setDiffuse(0.75);
        actor.getProperty().setSpecular(0.15);
      }
      viewport3D.render();
    };
    resetSegmentation3D = () => {
      opacity.value = "1";
      applyMaterials();
      if (labels.length && !surfaceError) {
        viewport3D.setCamera({ viewPlaneNormal: [0, 0, 1], viewUp: [0, -1, 0] });
        viewport3D.resetCamera(); // only visible prediction surfaces contribute bounds
        const camera = viewport3D.getCamera();
        if (camera.parallelScale) {
          const bounds = labels.filter((label) => selectedLabel === null || label === selectedLabel)
            .map((label) => (getSurfaceActorEntry(volume3dId, id, label)!.actor as vtkActor).getBounds());
          const extent = [0, 1, 2].map((axis) => Math.max(...bounds.map((b) => b[axis * 2 + 1]!)) - Math.min(...bounds.map((b) => b[axis * 2]!)));
          viewport3D.setCamera({ parallelScale: Math.max(camera.parallelScale, Math.hypot(...extent) * 0.55) });
        }
      }
    };
    resetSegmentation3D();
    addOverlayControls(shell.toolbar, mprIds, id, labels, args, (selected) => {
      selectedLabel = selected;
      for (const label of labels) segmentation.config.visibility.setSegmentIndexVisibility(volume3dId,
        { segmentationId: id, type: ToolEnums.SegmentationRepresentations.Surface }, label, selected === null || label === selected);
      applyMaterials();
    });
    const opacityLabel = element("label", "medcl-control", "3D 不透明度 ");
    opacityLabel.append(opacity); opacity.disabled = labels.length === 0 || !!surfaceError;
    opacity.addEventListener("input", applyMaterials);
    shell.toolbar.append(opacityLabel);
    args.setStateValue("selected_segment", null);
  } else {
    const applyMode = (mode: LayerMode, opacity: number) =>
      setRegistrationVolumes(mode, opacity, engine, mprIds, volume3dId, ids);
    const applyOpacity = (mode: LayerMode, opacity: number) =>
      setRegistrationOpacity(mode, opacity, engine, mprIds, ids);
    const initial = addRegistrationControls(shell.toolbar, envelope, args, applyMode, applyOpacity);
    await applyMode(initial, 0.5);
    args.setStateValue("layer_mode", initial);
    const warpedId = ids.get("warped_prediction");
    if (warpedId) {
      segmentationId = `${unique}-warped-segmentation`;
      segmentation.addSegmentations([{
        segmentationId,
        representation: { type: ToolEnums.SegmentationRepresentations.Labelmap, data: { volumeId: warpedId } },
      }]);
      for (const viewportId of [...mprIds, volume3dId]) {
        segmentation.addSegmentationRepresentations(viewportId, [{ segmentationId, type: ToolEnums.SegmentationRepresentations.Labelmap }]);
      }
      segmentation.config.style.setStyle(
        { segmentationId, type: ToolEnums.SegmentationRepresentations.Labelmap },
        { fillAlpha: 0.4, fillAlphaInactive: 0.4, renderOutline: true },
        true,
      );
      const id = segmentationId;
      bag.add(() => {
        for (const viewportId of [...mprIds, volume3dId]) {
          try { segmentation.removeSegmentationRepresentations(viewportId, { segmentationId: id }, true); } catch { /* already gone */ }
        }
        try { segmentation.removeSegmentation(id); } catch { /* already gone */ }
      });
      addOverlayControls(shell.toolbar, [...mprIds, volume3dId], segmentationId, envelope.segments, args);
    }
  }

  const sliceHandler = (): void => {
    for (let index = 0; index < mprIds.length; index += 1) {
      const viewport = engine.getViewport<VolumeViewport>(mprIds[index]!);
      shell.sliceLabels.get(viewportSpecs[index]![0])!.textContent = `slice ${viewport.getSliceIndex() + 1} / ${viewport.getNumberOfSlices()}`;
    }
  };
  for (const [index, viewportId] of mprIds.entries()) {
    const pane = shell.panes.get(viewportSpecs[index]![0])!;
    pane.addEventListener(CoreEnums.Events.IMAGE_RENDERED, sliceHandler);
    bag.add(() => pane.removeEventListener(CoreEnums.Events.IMAGE_RENDERED, sliceHandler));
  }
  const refresh = (): void => {
    if (cancelled()) return;
    engine.resize(true, envelope.viewerMode === "registration");
    engine.render();
  };
  const observer = new ResizeObserver(refresh);
  observer.observe(shell.grid);
  bag.add(() => observer.disconnect());
  const visibility = new IntersectionObserver((entries) => {
    if (entries.some((entry) => entry.isIntersecting)) refresh();
  });
  visibility.observe(shell.grid);
  bag.add(() => visibility.disconnect());
  const resizeFrame = requestAnimationFrame(refresh);
  bag.add(() => cancelAnimationFrame(resizeFrame));
  refresh();
  sliceHandler();
  shell.status.textContent = surfaceError ? "预测三维显示失败；二维切面可继续使用。" : envelope.viewerMode === "registration" && !ids.has("registered")
    ? "No registered volume was submitted; showing Fixed + Moving. TRE still uses predicted landmarks."
    : envelope.viewerMode === "segmentation" && !presentLabels(envelope.volumes.find((block) => block.name === "prediction")!).length
      ? "当前预览没有可显示的预测前景；切面原图仍可浏览。"
      : "Ready · wheel scrolls MPR slices · right drag zooms · middle drag pans";
  args.setStateValue("viewer_error_code", surfaceError ?? null);
  args.setStateValue("viewer_ready", !surfaceError);

}
