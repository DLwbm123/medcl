"""Validated raw-byte bridge for the read-only MedCL Cornerstone viewer."""

from __future__ import annotations

import json
import math
import re
import struct
from typing import Iterable

import numpy as np

ENVELOPE_SCHEMA = "medcl.cornerstone-envelope.v1"
_MAX_HEADER = 64 * 1024
_MAX_BYTES = 32 * 1024 * 1024
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,63}")
_IDENTITY_DIRECTION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
_DTYPES = {"uint8": np.dtype("u1"), "uint16": np.dtype("<u2"), "float32": np.dtype("<f4")}
_MODE_NAMES = {
    "segmentation": (("image", "scalar"), ("prediction", "labelmap")),
    "registration": (("fixed", "scalar"), ("moving", "scalar"), ("registered", "scalar"),
                     ("warped_prediction", "labelmap")),
}


def _numbers(value: object, length: int, name: str, *, positive: bool = False) -> list[float]:
    try:
        valid = (isinstance(value, (list, tuple, np.ndarray)) and len(value) == length and all(
            not isinstance(item, (bool, np.bool_)) and math.isfinite(float(item)) and (not positive or float(item) > 0)
            for item in value))
    except (OverflowError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"{name} 无效")
    return [float(item) for item in value]


def _context(value: object) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict) or not set(value) <= {"stage", "task_id", "case_id", "score", "downsampled"}:
        raise ValueError("viewer context 无效")
    output = {}
    for key in ("task_id", "case_id"):
        if key in value:
            if not isinstance(value[key], str) or _ID.fullmatch(value[key]) is None:
                raise ValueError("viewer ID 无效")
            output[key] = value[key]
    if "stage" in value:
        if type(value["stage"]) is not int or not 1 <= value["stage"] <= 12:
            raise ValueError("viewer stage 无效")
        output["stage"] = value["stage"]
    if "score" in value:
        score = value["score"]
        if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)):
            raise ValueError("viewer score 无效")
        output["score"] = score
    if "downsampled" in value:
        if type(value["downsampled"]) is not bool:
            raise ValueError("viewer downsampled 无效")
        output["downsampled"] = value["downsampled"]
    return output


def pack_envelope(*, viewer_mode: str, volumes: Iterable[tuple[str, str, np.ndarray]],
                  spacing_zyx: Iterable[float], spacing_source: str,
                  origin_xyz: Iterable[float] = (0, 0, 0),
                  direction_xyz: Iterable[float] = (1, 0, 0, 0, 1, 0, 0, 0, 1),
                  coordinate_mode: str = "index-space", segments: Iterable[int] = (),
                  context: dict | None = None) -> bytes:
    """Pack C-order Z/Y/X arrays behind one strict JSON header."""
    if viewer_mode not in _MODE_NAMES or coordinate_mode not in ("index-space", "fixed-display-grid"):
        raise ValueError("viewer mode 无效")
    if spacing_source not in ("protocol", "index-space-default"):
        raise ValueError("spacing source 无效")
    spacing = _numbers(spacing_zyx, 3, "spacing", positive=True)
    origin = _numbers(origin_xyz, 3, "origin")
    direction = _numbers(direction_xyz, 9, "direction")
    if any(abs(value - expected) > 1e-6 for value, expected in zip(direction, _IDENTITY_DIRECTION)):
        raise ValueError("当前 viewer 只支持 identity direction")
    declared, payloads, offset = [], [], 0
    allowed = dict(_MODE_NAMES[viewer_mode])
    for name, role, raw in volumes:
        if name not in allowed or role != allowed[name] or any(item["name"] == name for item in declared):
            raise ValueError("volume name / role 无效或重复")
        array = np.asarray(raw)
        if array.ndim != 3 or any(size <= 0 for size in array.shape) or not array.flags.c_contiguous:
            raise ValueError("volume 必须是连续非空 Z/Y/X 数组")
        dtype_name = next((key for key, dtype in _DTYPES.items() if array.dtype == dtype), None)
        if dtype_name is None or (role == "labelmap" and dtype_name == "float32"):
            raise ValueError("volume dtype 不在白名单")
        if role == "scalar" and not np.isfinite(array).all():
            raise ValueError("scalar volume 包含非有限值")
        payload = array.tobytes(order="C")
        declared.append({"name": name, "role": role, "dtype": dtype_name,
                         "shape_zyx": [int(size) for size in array.shape], "offset": offset,
                         "byte_length": len(payload)})
        payloads.append(payload)
        offset += len(payload)
    names = [item["name"] for item in declared]
    if viewer_mode == "segmentation" and names != ["image", "prediction"]:
        raise ValueError("分割 envelope 必须依次包含 image 和 prediction")
    if viewer_mode == "registration" and (names[:2] != ["fixed", "moving"] or names[2:] not in (
            [], ["registered"], ["warped_prediction"], ["registered", "warped_prediction"])):
        raise ValueError("配准 envelope volume 顺序无效")
    if not declared:
        raise ValueError("viewer envelope 没有 volume")
    if any(item["shape_zyx"] != declared[0]["shape_zyx"] for item in declared[1:]):
        raise ValueError("viewer volumes 必须使用同一显示网格")
    segment_list = list(segments)
    if (len(segment_list) != len(set(segment_list)) or any(type(item) is not int or not 1 <= item <= 65535 for item in segment_list)):
        raise ValueError("segments 无效")
    header = {"schema": ENVELOPE_SCHEMA, "viewer_mode": viewer_mode, "coordinate_mode": coordinate_mode,
              "spacing_source": spacing_source, "spacing_zyx": spacing, "origin_xyz": origin,
              "direction_xyz": direction, "volumes": declared, "segments": segment_list,
              "context": _context(context)}
    encoded = json.dumps(header, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if not 0 < len(encoded) <= _MAX_HEADER or 4 + len(encoded) + offset > _MAX_BYTES:
        raise ValueError("viewer envelope 超过大小上限")
    return struct.pack("<I", len(encoded)) + encoded + b"".join(payloads)


def unpack_envelope(data: bytes) -> tuple[dict, dict[str, np.ndarray]]:
    """Reference decoder used by Python tests; the browser validates independently."""
    if not isinstance(data, bytes) or not 4 < len(data) <= _MAX_BYTES:
        raise ValueError("viewer envelope 长度无效")
    header_length = struct.unpack("<I", data[:4])[0]
    if not 0 < header_length <= _MAX_HEADER or 4 + header_length > len(data):
        raise ValueError("viewer envelope header 截断")
    try:
        header = json.loads(data[4:4 + header_length])
    except (UnicodeError, json.JSONDecodeError):
        raise ValueError("viewer envelope header 无效") from None
    required = {"schema", "viewer_mode", "coordinate_mode", "spacing_source", "spacing_zyx", "origin_xyz",
                "direction_xyz", "volumes", "segments", "context"}
    if not isinstance(header, dict) or set(header) != required or header.get("schema") != ENVELOPE_SCHEMA:
        raise ValueError("viewer envelope schema 无效")
    volumes = header.get("volumes")
    if not isinstance(volumes, list) or not 1 <= len(volumes) <= 4:
        raise ValueError("viewer volume 数量无效")
    payload = memoryview(data)[4 + header_length:]
    decoded, cursor = {}, 0
    for item in volumes:
        fields = {"name", "role", "dtype", "shape_zyx", "offset", "byte_length"}
        if not isinstance(item, dict) or set(item) != fields or item.get("dtype") not in _DTYPES:
            raise ValueError("viewer volume header 无效")
        shape = item.get("shape_zyx")
        if (not isinstance(shape, list) or len(shape) != 3 or any(type(size) is not int or size <= 0 for size in shape)
                or type(item.get("offset")) is not int or type(item.get("byte_length")) is not int):
            raise ValueError("viewer volume shape / offset 无效")
        expected = math.prod(shape) * _DTYPES[item["dtype"]].itemsize
        if item["offset"] != cursor or item["byte_length"] != expected or cursor + expected > len(payload):
            raise ValueError("viewer volume 数据有空洞、重叠或截断")
        decoded[item["name"]] = np.frombuffer(payload[cursor:cursor + expected], dtype=_DTYPES[item["dtype"]]).reshape(shape)
        cursor += expected
    if cursor != len(payload):
        raise ValueError("viewer envelope 包含多余尾部数据")
    # Reuse the writer's full semantic validation without trusting the decoded header.
    rebuilt = pack_envelope(viewer_mode=header["viewer_mode"],
                            volumes=[(item["name"], item["role"], decoded[item["name"]]) for item in volumes],
                            spacing_zyx=header["spacing_zyx"], spacing_source=header["spacing_source"],
                            origin_xyz=header["origin_xyz"], direction_xyz=header["direction_xyz"],
                            coordinate_mode=header["coordinate_mode"], segments=header["segments"], context=header["context"])
    if len(rebuilt) != len(data):
        raise ValueError("viewer envelope 语义无效")
    return header, decoded


def envelope_from_preview(reference: dict, arrays: dict[str, np.ndarray]) -> bytes:
    """Convert an already validated private preview into the one-case browser envelope."""
    context = {key: reference.get(key) for key in ("stage", "task_id", "case_id", "score", "downsampled") if key in reference}
    spacing_source = str(arrays["spacing_source"])
    if reference["kind"] == "segmentation-volume":
        prediction = arrays["prediction_volume"]
        return pack_envelope(viewer_mode="segmentation",
                             volumes=[("image", "scalar", arrays["image_volume"]),
                                      ("prediction", "labelmap", prediction)],
                             spacing_zyx=arrays["spacing_zyx"], spacing_source=spacing_source,
                             segments=sorted(int(item) for item in np.unique(prediction) if item > 0), context=context)
    if reference["kind"] != "registration-volume":
        raise ValueError("该预览不是三维 Cornerstone 格式")
    volumes = [("fixed", "scalar", arrays["fixed_volume"]), ("moving", "scalar", arrays["moving_volume"])]
    if "registered_volume" in arrays:
        volumes.append(("registered", "scalar", arrays["registered_volume"]))
    if "warped_prediction" in arrays:
        volumes.append(("warped_prediction", "labelmap", arrays["warped_prediction"]))
    labelmap = arrays.get("warped_prediction")
    return pack_envelope(viewer_mode="registration", volumes=volumes,
                         spacing_zyx=arrays["spacing_zyx"], spacing_source=spacing_source,
                         origin_xyz=arrays["origin_xyz"], direction_xyz=arrays["direction_xyz"],
                         coordinate_mode=reference.get("coordinate_mode", "index-space"),
                         segments=[] if labelmap is None else sorted(int(item) for item in np.unique(labelmap) if item > 0),
                         context=context)


def render(envelope: bytes, *, key: str):
    """Mount the installed package-based Streamlit Components v2 viewer."""
    from streamlit.components.v2 import component
    viewer = component("medcl-cornerstone.viewer", js="index.js")
    return viewer(data=envelope, key=key, height=760, width="stretch",
                  default={"viewer_ready": False, "viewer_error_code": None, "selected_segment": None,
                           "layer_mode": None},
                  on_viewer_ready_change=lambda: None, on_viewer_error_code_change=lambda: None,
                  on_selected_segment_change=lambda: None, on_layer_mode_change=lambda: None)
