import {
  Enums as CoreEnums,
  RenderingEngine,
  Types,
  cache,
  init as initCore,
  volumeLoader,
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
import { CleanupBag } from "./lifecycle";
import viewerStyles from "./styles.css?inline";
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

const initialize = (): Promise<void> => {
  if (!initialized) {
    initialized = Promise.resolve().then(() => {
      initCore();
      initTools();
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
  const options = {
    metadata: metadata(dimensions, spacing, frame, block.dtype),
    dimensions,
    spacing,
    origin: envelope.originXYZ,
    direction: envelope.directionXYZ,
    scalarData: block.data as Types.PixelDataTypedArray,
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
  mpr.setToolActive(WindowLevelTool.toolName, { bindings: [{ mouseButton: ToolEnums.MouseBindings.Primary }] });
  mpr.setToolActive(PanTool.toolName, { bindings: [{ mouseButton: ToolEnums.MouseBindings.Auxiliary }] });
  mpr.setToolActive(ZoomTool.toolName, { bindings: [{ mouseButton: ToolEnums.MouseBindings.Secondary }] });
  mpr.setToolActive(StackScrollTool.toolName, { bindings: [{ mouseButton: ToolEnums.MouseBindings.Wheel }] });
  volume3d.setToolActive(TrackballRotateTool.toolName, { bindings: [{ mouseButton: ToolEnums.MouseBindings.Primary }] });
  volume3d.setToolActive(PanTool.toolName, { bindings: [{ mouseButton: ToolEnums.MouseBindings.Auxiliary }] });
  volume3d.setToolActive(ZoomTool.toolName, { bindings: [
    { mouseButton: ToolEnums.MouseBindings.Secondary },
    { mouseButton: ToolEnums.MouseBindings.Wheel },
  ] });
  return [mprGroupId, volumeGroupId];
};

const activateMprTool = (groupId: string, name: string): void => {
  const group = ToolGroupManager.getToolGroup(groupId);
  if (!group) return;
  group.setToolActive(name, { bindings: [{ mouseButton: ToolEnums.MouseBindings.Primary }] });
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
  for (const [id, label] of [...viewportSpecs.map(([id, label]) => [id, label] as const), ["volume3d", "3D volume"] as const]) {
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

const addToolButtons = (toolbar: HTMLDivElement, mprGroupId: string, engine: RenderingEngine): void => {
  const actions: Array<[string, string, string]> = [
    ["W/L", "Window / level", WindowLevelTool.toolName],
    ["Pan", "Pan all MPR views", PanTool.toolName],
    ["Zoom", "Zoom all MPR views", ZoomTool.toolName],
    ["Crosshair", "Linked crosshair navigation", CrosshairsTool.toolName],
  ];
  for (const [label, title, tool] of actions) {
    const control = button(label, title);
    control.addEventListener("click", () => activateMprTool(mprGroupId, tool));
    toolbar.append(control);
  }
  const reset = button("Reset", "Reset cameras and display properties");
  reset.addEventListener("click", () => {
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
): void => {
  const visibleLabel = element("label", "medcl-control");
  const visible = element("input") as HTMLInputElement;
  visible.type = "checkbox";
  visible.checked = true;
  visibleLabel.append(visible, document.createTextNode(" overlay"));
  visible.addEventListener("change", () => {
    for (const viewportId of viewportIds) {
      segmentation.config.visibility.setSegmentationRepresentationVisibility(
        viewportId,
        { segmentationId, type: ToolEnums.SegmentationRepresentations.Labelmap },
        visible.checked,
      );
    }
  });
  const opacityLabel = element("label", "medcl-control", "Opacity ");
  const opacity = element("input") as HTMLInputElement;
  opacity.type = "range";
  opacity.min = "0";
  opacity.max = "1";
  opacity.step = "0.05";
  opacity.value = "0.45";
  opacity.addEventListener("input", () => {
    segmentation.config.style.setStyle(
      { segmentationId, type: ToolEnums.SegmentationRepresentations.Labelmap },
      { fillAlpha: Number(opacity.value), fillAlphaInactive: Number(opacity.value), renderOutline: true },
      true,
    );
  });
  opacityLabel.append(opacity);
  toolbar.append(visibleLabel, opacityLabel);
  if (segments.length) {
    const segmentLabel = element("label", "medcl-control", "Segment ");
    const select = element("select") as HTMLSelectElement;
    const all = element("option", "", "All");
    all.value = "all";
    select.append(all);
    for (const segment of segments) {
      const option = element("option", "", `Label ${segment}`);
      option.value = String(segment);
      select.append(option);
    }
    select.addEventListener("change", () => {
      const selected = select.value === "all" ? null : Number(select.value);
      for (const viewportId of viewportIds) {
        for (const segment of segments) {
          segmentation.config.visibility.setSegmentIndexVisibility(
            viewportId,
            { segmentationId, type: ToolEnums.SegmentationRepresentations.Labelmap },
            segment,
            selected === null || selected === segment,
          );
        }
      }
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

const addRegistrationControls = (
  toolbar: HTMLDivElement,
  envelope: ParsedEnvelope,
  args: Args,
  apply: (mode: LayerMode, opacity: number) => Promise<void>,
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
  const update = (): void => {
    const mode = select.value as LayerMode;
    args.setStateValue("layer_mode", mode);
    void apply(mode, Number(opacity.value)).catch(() => args.setStateValue("viewer_error_code", "LAYER_UPDATE_FAILED"));
  };
  select.addEventListener("change", update);
  opacity.addEventListener("input", update);
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
  for (const block of envelope.volumes) {
    const volumeId = `${unique}-${block.name}`;
    ids.set(block.name, volumeId);
    imageIds.set(volumeId, createVolume(block, volumeId, envelope, frame));
  }
  bag.add(() => {
    for (const [volumeId, slices] of imageIds) {
      try { cache.removeVolumeLoadObject(volumeId); } catch { /* already evicted */ }
      for (const imageId of slices) {
        try { cache.removeImageLoadObject(imageId, { force: true }); } catch { /* already evicted */ }
      }
    }
  });
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
  addToolButtons(shell.toolbar, groupIds[0]!, engine);

  const syncs = [
    synchronizers.createZoomPanSynchronizer(`${unique}-zoom-pan`),
    synchronizers.createVOISynchronizer(`${unique}-voi`, { syncInvertState: true, syncColormap: false }),
  ];
  bag.add(() => { for (const sync of syncs) sync.destroy(); });
  for (const sync of syncs) for (const viewportId of mprIds) sync.add({ renderingEngineId: engineId, viewportId });

  let segmentationId: string | undefined;
  if (envelope.viewerMode === "segmentation") {
    const scalarId = ids.get("image")!;
    for (const viewportId of [...mprIds, volume3dId]) {
      const viewport = engine.getViewport<VolumeViewport>(viewportId);
      await viewport.setVolumes([{ volumeId: scalarId, actorUID: scalarId }], false);
      viewport.setProperties({ voiRange: normalizedVoi }, scalarId);
      viewport.resetCamera();
    }
    engine.getViewport<VolumeViewport>(volume3dId).setProperties({ preset: "MR-Default" }, scalarId);
    const imageLabel = element("label", "medcl-control");
    const imageVisible = element("input") as HTMLInputElement;
    imageVisible.type = "checkbox";
    imageVisible.checked = true;
    imageLabel.append(imageVisible, document.createTextNode(" image"));
    imageVisible.addEventListener("change", () => {
      for (const viewportId of [...mprIds, volume3dId]) {
        engine.getViewport<VolumeViewport>(viewportId).getActor(scalarId)?.actor.setVisibility(imageVisible.checked);
      }
      engine.render();
    });
    shell.toolbar.append(imageLabel);
    segmentationId = `${unique}-prediction-segmentation`;
    segmentation.addSegmentations([{
      segmentationId,
      representation: {
        type: ToolEnums.SegmentationRepresentations.Labelmap,
        data: { volumeId: ids.get("prediction")! },
      },
    }]);
    for (const viewportId of [...mprIds, volume3dId]) {
      segmentation.addSegmentationRepresentations(viewportId, [{
        segmentationId,
        type: ToolEnums.SegmentationRepresentations.Labelmap,
      }]);
    }
    segmentation.config.style.setStyle(
      { segmentationId, type: ToolEnums.SegmentationRepresentations.Labelmap },
      { fillAlpha: 0.45, fillAlphaInactive: 0.45, renderOutline: true, outlineWidth: 1 },
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
  } else {
    const apply = (mode: LayerMode, opacity: number) =>
      setRegistrationVolumes(mode, opacity, engine, mprIds, volume3dId, ids);
    const initial = addRegistrationControls(shell.toolbar, envelope, args, apply);
    await apply(initial, 0.5);
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
    engine.resize(true, true);
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
  shell.status.textContent = envelope.viewerMode === "registration" && !ids.has("registered")
    ? "No registered volume was submitted; showing Fixed + Moving. TRE still uses predicted landmarks."
    : envelope.viewerMode === "segmentation" && envelope.segments.length === 0
      ? "Prediction is empty; the image is available but no foreground overlay or 3D labelmap exists."
      : "Ready · wheel scrolls MPR slices · right drag zooms · middle drag pans";
  args.setStateValue("viewer_error_code", null);
  args.setStateValue("viewer_ready", true);

}
