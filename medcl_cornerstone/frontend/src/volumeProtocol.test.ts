import { describe, expect, it } from "vitest";
import { defaultRegistrationLayer, parseEnvelope, registrationLayerChoices, stableId, xyzDimensions, xyzSpacing } from "./volumeProtocol";

const encode = (header: Record<string, unknown>, payload: Uint8Array): Uint8Array => {
  const json = new TextEncoder().encode(JSON.stringify(header));
  const data = new Uint8Array(4 + json.length + payload.length);
  new DataView(data.buffer).setUint32(0, json.length, true);
  data.set(json, 4);
  data.set(payload, 4 + json.length);
  return data;
};

const header = (volumes: unknown[]) => ({
  schema: "medcl.cornerstone-envelope.v1",
  viewer_mode: "segmentation",
  coordinate_mode: "index-space",
  spacing_source: "protocol",
  spacing_zyx: [3, 2, 1],
  origin_xyz: [0, 0, 0],
  direction_xyz: [1, 0, 0, 0, 1, 0, 0, 0, 1],
  volumes,
  segments: [1],
  context: { stage: 1, task_id: "task-1", downsampled: false },
});

describe("volume envelope", () => {
  it("decodes strict C-order blocks and converts ZYX geometry", () => {
    const volumes = [
      { name: "image", role: "scalar", dtype: "uint8", shape_zyx: [1, 2, 2], offset: 0, byte_length: 4 },
      { name: "prediction", role: "labelmap", dtype: "uint16", shape_zyx: [1, 2, 2], offset: 4, byte_length: 8 },
    ];
    const parsed = parseEnvelope(encode(header(volumes), new Uint8Array([0, 1, 2, 3, 1, 0, 0, 0, 1, 0, 0, 0])));
    expect(Array.from(parsed.volumes[1]!.data)).toEqual([1, 0, 1, 0]);
    expect(xyzDimensions(parsed.volumes[0]!.shapeZYX)).toEqual([2, 2, 1]);
    expect(xyzSpacing(parsed.spacingZYX)).toEqual([1, 2, 3]);
  });

  it("accepts only identity direction within a tiny tolerance", () => {
    const volumes = [
      { name: "image", role: "scalar", dtype: "uint8", shape_zyx: [1, 1, 1], offset: 0, byte_length: 1 },
      { name: "prediction", role: "labelmap", dtype: "uint8", shape_zyx: [1, 1, 1], offset: 1, byte_length: 1 },
    ];
    expect(parseEnvelope(encode({ ...header(volumes), direction_xyz: [1 + 5e-7, 0, 0, 0, 1, 0, 0, 0, 1] }, new Uint8Array(2))).directionXYZ[0]).toBeCloseTo(1);
    expect(() => parseEnvelope(encode({ ...header(volumes), direction_xyz: [-1, 0, 0, 0, 1, 0, 0, 0, 1] }, new Uint8Array(2)))).toThrow("unsupported_direction");
  });

  it.each([
    ["gap", 1, 8],
    ["overlap", 3, 8],
    ["wrong length", 4, 6],
  ])("rejects %s in payload layout", (_name, offset, length) => {
    const volumes = [
      { name: "image", role: "scalar", dtype: "uint8", shape_zyx: [1, 2, 2], offset: 0, byte_length: 4 },
      { name: "prediction", role: "labelmap", dtype: "uint16", shape_zyx: [1, 2, 2], offset, byte_length: length },
    ];
    expect(() => parseEnvelope(encode(header(volumes), new Uint8Array(12)))).toThrow();
  });

  it("rejects mismatched grids, trailing bytes, and unexpected fields", () => {
    const volumes = [
      { name: "image", role: "scalar", dtype: "uint8", shape_zyx: [1, 2, 2], offset: 0, byte_length: 4 },
      { name: "prediction", role: "labelmap", dtype: "uint16", shape_zyx: [2, 1, 2], offset: 4, byte_length: 8 },
    ];
    expect(() => parseEnvelope(encode(header(volumes), new Uint8Array(12)))).toThrow("grid_mismatch");
    volumes[1]!.shape_zyx = [1, 2, 2];
    expect(() => parseEnvelope(encode(header(volumes), new Uint8Array(13)))).toThrow("trailing_payload");
    expect(() => parseEnvelope(encode({ ...header(volumes), secret: "x" }, new Uint8Array(12)))).toThrow("invalid_header_fields");
  });

  it("rejects truncated headers, truncated payloads, and labelmap floats", () => {
    expect(() => parseEnvelope(new Uint8Array([8, 0, 0, 0, 123]))).toThrow("truncated_header");
    const volumes = [
      { name: "image", role: "scalar", dtype: "uint8", shape_zyx: [1, 2, 2], offset: 0, byte_length: 4 },
      { name: "prediction", role: "labelmap", dtype: "uint16", shape_zyx: [1, 2, 2], offset: 4, byte_length: 8 },
    ];
    expect(() => parseEnvelope(encode(header(volumes), new Uint8Array(11)))).toThrow("invalid_payload_layout");
    volumes[1] = { ...volumes[1]!, dtype: "float32", byte_length: 16 };
    expect(() => parseEnvelope(encode(header(volumes), new Uint8Array(20)))).toThrow("invalid_dtype");
  });

  it("exposes only available registration layers and a safe fallback", () => {
    const parsed = { viewerMode: "registration", volumes: [{ name: "fixed" }, { name: "moving" }] } as never;
    expect(registrationLayerChoices(parsed).map((choice) => choice.value)).toEqual(["fixed", "moving", "fixed-moving"]);
    expect(defaultRegistrationLayer(parsed)).toBe("fixed-moving");
    const withRegistered = { viewerMode: "registration", volumes: [{ name: "fixed" }, { name: "moving" }, { name: "registered" }] } as never;
    expect(registrationLayerChoices(withRegistered).map((choice) => choice.value)).toContain("fixed-registered");
    expect(defaultRegistrationLayer(withRegistered)).toBe("fixed-registered");
    expect(stableId("volume", "same")).toBe(stableId("volume", "same"));
    expect(stableId("volume", "same")).not.toBe(stableId("volume", "different"));
  });
});
