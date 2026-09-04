export const ENVELOPE_SCHEMA = "medcl.cornerstone-envelope.v1";
const MAX_HEADER = 64 * 1024;
const MAX_BYTES = 32 * 1024 * 1024;
const ID = /^[A-Za-z0-9][A-Za-z0-9-]{0,63}$/;

export type ViewerMode = "segmentation" | "registration";
export type DType = "uint8" | "uint16" | "float32";
export type VolumeName = "image" | "prediction" | "fixed" | "moving" | "registered" | "warped_prediction";
export type VolumeRole = "scalar" | "labelmap";
export type LayerMode = "fixed" | "moving" | "registered" | "fixed-moving" | "fixed-registered";

export interface ViewerContext {
  stage?: number;
  task_id?: string;
  case_id?: string;
  score?: number | null;
  downsampled?: boolean;
}

export interface VolumeBlock {
  name: VolumeName;
  role: VolumeRole;
  dtype: DType;
  shapeZYX: [number, number, number];
  data: Uint8Array | Uint16Array | Float32Array;
}

export interface ParsedEnvelope {
  viewerMode: ViewerMode;
  coordinateMode: "index-space" | "fixed-display-grid";
  spacingSource: "protocol" | "index-space-default";
  spacingZYX: [number, number, number];
  originXYZ: [number, number, number];
  directionXYZ: [number, number, number, number, number, number, number, number, number];
  segments: number[];
  context: ViewerContext;
  volumes: VolumeBlock[];
}

type HeaderVolume = {
  name: unknown;
  role: unknown;
  dtype: unknown;
  shape_zyx: unknown;
  offset: unknown;
  byte_length: unknown;
};

const exactKeys = (value: object, keys: string[]): boolean => {
  const actual = Object.keys(value).sort();
  return actual.length === keys.length && actual.every((item, index) => item === [...keys].sort()[index]);
};

const tuple = <N extends number>(value: unknown, length: N, positive = false): number[] => {
  if (!Array.isArray(value) || value.length !== length || value.some((item) =>
    typeof item !== "number" || !Number.isFinite(item) || (positive && item <= 0))) {
    throw new Error("invalid_geometry");
  }
  return value;
};

const uint = (value: unknown, min: number, max: number): number => {
  if (!Number.isInteger(value) || (value as number) < min || (value as number) > max) {
    throw new Error("invalid_integer");
  }
  return value as number;
};

const validVolumeOrder = (mode: ViewerMode, items: HeaderVolume[]): boolean => {
  const pairs = items.map((item) => `${item.name}:${item.role}`);
  if (mode === "segmentation") return pairs.join(",") === "image:scalar,prediction:labelmap";
  return [
    "fixed:scalar,moving:scalar",
    "fixed:scalar,moving:scalar,registered:scalar",
    "fixed:scalar,moving:scalar,warped_prediction:labelmap",
    "fixed:scalar,moving:scalar,registered:scalar,warped_prediction:labelmap",
  ].includes(pairs.join(","));
};

const parseContext = (value: unknown): ViewerContext => {
  const keys = ["stage", "task_id", "case_id", "score", "downsampled"];
  if (!value || typeof value !== "object" || Object.keys(value).some((key) => !keys.includes(key))) {
    throw new Error("invalid_context");
  }
  const raw = value as Record<string, unknown>;
  const context: ViewerContext = {};
  if (raw.stage !== undefined) context.stage = uint(raw.stage, 1, 12);
  for (const key of ["task_id", "case_id"] as const) {
    if (raw[key] !== undefined) {
      if (typeof raw[key] !== "string" || !ID.test(raw[key])) throw new Error("invalid_context_id");
      context[key] = raw[key];
    }
  }
  if (raw.score !== undefined) {
    if (raw.score !== null && (typeof raw.score !== "number" || !Number.isFinite(raw.score))) throw new Error("invalid_score");
    context.score = raw.score as number | null;
  }
  if (raw.downsampled !== undefined) {
    if (typeof raw.downsampled !== "boolean") throw new Error("invalid_downsampled");
    context.downsampled = raw.downsampled;
  }
  return context;
};

export function parseEnvelope(input: Uint8Array | ArrayBuffer): ParsedEnvelope {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  if (bytes.byteLength <= 4 || bytes.byteLength > MAX_BYTES) throw new Error("invalid_envelope_length");
  const headerLength = new DataView(bytes.buffer, bytes.byteOffset, 4).getUint32(0, true);
  if (headerLength <= 0 || headerLength > MAX_HEADER || 4 + headerLength > bytes.byteLength) {
    throw new Error("truncated_header");
  }
  let raw: unknown;
  try {
    raw = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes.subarray(4, 4 + headerLength)));
  } catch {
    throw new Error("invalid_header_json");
  }
  const headerKeys = ["schema", "viewer_mode", "coordinate_mode", "spacing_source", "spacing_zyx", "origin_xyz", "direction_xyz", "volumes", "segments", "context"];
  if (!raw || typeof raw !== "object" || !exactKeys(raw, headerKeys)) throw new Error("invalid_header_fields");
  const header = raw as Record<string, unknown>;
  if (header.schema !== ENVELOPE_SCHEMA || !["segmentation", "registration"].includes(String(header.viewer_mode))) {
    throw new Error("unsupported_schema_or_mode");
  }
  const viewerMode = header.viewer_mode as ViewerMode;
  if (!["index-space", "fixed-display-grid"].includes(String(header.coordinate_mode)) ||
      !["protocol", "index-space-default"].includes(String(header.spacing_source))) {
    throw new Error("invalid_coordinate_metadata");
  }
  const spacingZYX = tuple(header.spacing_zyx, 3, true) as ParsedEnvelope["spacingZYX"];
  const originXYZ = tuple(header.origin_xyz, 3) as ParsedEnvelope["originXYZ"];
  const directionXYZ = tuple(header.direction_xyz, 9) as ParsedEnvelope["directionXYZ"];
  if (!Array.isArray(header.segments) || header.segments.some((value) => !Number.isInteger(value) || value < 1 || value > 65535) ||
      new Set(header.segments).size !== header.segments.length) throw new Error("invalid_segments");
  if (!Array.isArray(header.volumes) || header.volumes.length < 1 || header.volumes.length > 4) throw new Error("invalid_volume_count");
  const declared = header.volumes as HeaderVolume[];
  if (declared.some((item) => !item || typeof item !== "object" ||
      !exactKeys(item as object, ["name", "role", "dtype", "shape_zyx", "offset", "byte_length"]))) {
    throw new Error("invalid_volume_header");
  }
  if (!validVolumeOrder(viewerMode, declared)) throw new Error("invalid_volume_order");
  const payloadStart = 4 + headerLength;
  const payloadLength = bytes.byteLength - payloadStart;
  let cursor = 0;
  let shape: number[] | undefined;
  const volumes: VolumeBlock[] = declared.map((item) => {
    const dtype = item.dtype as DType;
    if (!["uint8", "uint16", "float32"].includes(String(dtype)) ||
        (item.role === "labelmap" && dtype === "float32")) throw new Error("invalid_dtype");
    const shapeZYX = tuple(item.shape_zyx, 3, true);
    if (shapeZYX.some((size) => !Number.isInteger(size))) throw new Error("invalid_shape");
    if (shape && shape.some((size, index) => size !== shapeZYX[index])) throw new Error("grid_mismatch");
    shape = shapeZYX;
    const size = dtype === "uint8" ? 1 : dtype === "uint16" ? 2 : 4;
    const expected = shapeZYX.reduce((total, value) => total * value, 1) * size;
    if (!Number.isSafeInteger(expected) || item.offset !== cursor || item.byte_length !== expected || cursor + expected > payloadLength) {
      throw new Error("invalid_payload_layout");
    }
    const copy = bytes.slice(payloadStart + cursor, payloadStart + cursor + expected).buffer;
    cursor += expected;
    const data = dtype === "uint8" ? new Uint8Array(copy) : dtype === "uint16" ? new Uint16Array(copy) : new Float32Array(copy);
    if (dtype === "float32") for (const value of data) if (!Number.isFinite(value)) throw new Error("non_finite_scalar");
    return { name: item.name as VolumeName, role: item.role as VolumeRole, dtype, shapeZYX: shapeZYX as [number, number, number], data };
  });
  if (cursor !== payloadLength) throw new Error("trailing_payload");
  return {
    viewerMode,
    coordinateMode: header.coordinate_mode as ParsedEnvelope["coordinateMode"],
    spacingSource: header.spacing_source as ParsedEnvelope["spacingSource"],
    spacingZYX,
    originXYZ,
    directionXYZ,
    segments: header.segments as number[],
    context: parseContext(header.context),
    volumes,
  };
}

export const xyzDimensions = ([z, y, x]: [number, number, number]): [number, number, number] => [x, y, z];
export const xyzSpacing = ([z, y, x]: [number, number, number]): [number, number, number] => [x, y, z];

export function stableId(prefix: string, key: string): string {
  let value = 2166136261;
  for (const char of key) value = Math.imul(value ^ char.charCodeAt(0), 16777619);
  return `${prefix}-${(value >>> 0).toString(16)}`;
}

export function registrationLayerChoices(envelope: ParsedEnvelope): Array<{ value: LayerMode; label: string }> {
  if (envelope.viewerMode !== "registration") return [];
  const names = new Set(envelope.volumes.map((volume) => volume.name));
  const choices: Array<{ value: LayerMode; label: string }> = [
    { value: "fixed", label: "Fixed only" },
    { value: "moving", label: "Moving only" },
    { value: "fixed-moving", label: "Fixed + Moving" },
  ];
  if (names.has("registered")) choices.push(
    { value: "registered", label: "Registered only" },
    { value: "fixed-registered", label: "Fixed + Registered" },
  );
  return choices;
}

export function defaultRegistrationLayer(envelope: ParsedEnvelope): LayerMode {
  return envelope.volumes.some((volume) => volume.name === "registered") ? "fixed-registered" : "fixed-moving";
}
