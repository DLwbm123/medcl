"""Bounded, non-executable uploads. Stage assignments always come from the UI."""

from __future__ import annotations

import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
import zipfile

import numpy as np

from medcl.benchmarks import freeze_assets, public_protocol, readiness
from medcl.storage import create_job

MAX_FILE = 128 * 1024 * 1024
MAX_TOTAL = 256 * 1024 * 1024
MAX_EXPANDED = 512 * 1024 * 1024
SCHEMA = "medcl.predictions.v1"
ARCHITECTURES = {
    "classification": {"linear-classifier-v1": "线性分类器 · 固定全局类别输出"},
    "segmentation": {"pixel-linear-v1": "逐像素线性分割器 · 共享输出头"},
    "registration": {"point-translation-v1": "标志点平移模型 · 固定空间毫米坐标"},
}


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON 包含重复键")
        result[key] = value
    return result


def parse_json(data: bytes):
    def bad_constant(_):
        raise ValueError("JSON 不允许 NaN / Infinity")
    try:
        return json.loads(data, object_pairs_hook=_pairs, parse_constant=bad_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        raise ValueError("JSON 文件无效或嵌套过深") from None


def validate_archive(data: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if not 2 <= len(entries) <= 24 or len({e.filename for e in entries}) != len(entries):
                raise ValueError("NPZ 必须有 1–12 个任务的成对数组，且不可重复")
            if sum(e.file_size for e in entries) > MAX_EXPANDED:
                raise ValueError("NPZ 解压后超过 512 MiB 上限")
            for entry in entries:
                path = PurePosixPath(entry.filename)
                if path.is_absolute() or len(path.parts) != 1 or not re.fullmatch(r"[A-Za-z0-9-]+__(ids|pred)\.npy", entry.filename):
                    raise ValueError("ZIP 路径非法：只允许 task__ids.npy 和 task__pred.npy")
                if (entry.external_attr >> 16) & 0o170000 == 0o120000 or entry.flag_bits & 1:
                    raise ValueError("不接受符号链接或加密 ZIP")
                if entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                    raise ValueError("不支持该 ZIP 压缩算法")
                with archive.open(entry) as handle:
                    version = np.lib.format.read_magic(handle)
                    if version == (1, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle, max_header_size=10000)
                    elif version == (2, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle, max_header_size=10000)
                    else:
                        raise ValueError("仅支持 NPY v1/v2")
                    if dtype.hasobject or dtype.kind not in "biufUS" or len(shape) > 5:
                        raise ValueError("不接受 Pickle、对象或结构化数组")
                    if not shape or any(d <= 0 for d in shape) or math.prod(shape) * dtype.itemsize > MAX_EXPANDED:
                        raise ValueError("数组维度为空或超过资源上限")
                    if handle.tell() + math.prod(shape) * dtype.itemsize != entry.file_size:
                        raise ValueError("数组声明大小与文件内容不符")
    except (zipfile.BadZipFile, EOFError, OSError, OverflowError):
        raise ValueError("NPZ/ZIP 内容损坏") from None


def validate_weights(data: bytes, architecture: str) -> None:
    if len(data) < 10:
        raise ValueError("safetensors 文件过短")
    length = struct.unpack("<Q", data[:8])[0]
    if not 2 <= length <= 1024 * 1024 or 8 + length > len(data):
        raise ValueError("safetensors 头部长度无效")
    header = parse_json(data[8:8 + length])
    if not isinstance(header, dict):
        raise ValueError("safetensors 头部必须为对象")
    keys = set(header) - {"__metadata__"}
    expected = {"offset"} if architecture == "point-translation-v1" else {"weight", "bias"}
    if keys != expected:
        raise ValueError("权重键与所选已审核结构不符")
    intervals = []
    for name in keys:
        item = header[name]
        if not isinstance(item, dict) or item.get("dtype") != "F32":
            raise ValueError("首版权重只接受 float32 纯张量")
        shape, offsets = item.get("shape"), item.get("data_offsets")
        if not isinstance(shape, list) or not 1 <= len(shape) <= 2 or any(type(d) is not int or not 0 < d <= 1048576 for d in shape):
            raise ValueError("权重维度无效")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(i) is not int or i < 0 for i in offsets):
            raise ValueError("权重字节偏移无效")
        if offsets[1] - offsets[0] != math.prod(shape) * 4:
            raise ValueError("权重大小与维度不符")
        intervals.append(tuple(offsets))
    end = 0
    for start, stop in sorted(intervals):
        if start != end or stop < start:
            raise ValueError("权重片段有空洞或重叠")
        end = stop
    if end != len(data) - 8 - length:
        raise ValueError("权重数据截断或存在多余内容")


def inspect_upload(name: str, data: bytes, mode: str, architecture: str | None = None) -> None:
    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_FILE:
        raise ValueError("每个文件需为 1 byte–128 MiB")
    suffix = Path(name).suffix.lower()
    if mode == "model":
        if suffix != ".safetensors":
            raise ValueError("模型仅接受 safetensors；禁止 Python、Pickle、PT/PTH")
        validate_weights(data, architecture)
    elif suffix in (".npz", ".zip"):
        validate_archive(data)
    elif suffix == ".json":
        doc = parse_json(data)
        if not isinstance(doc, dict) or doc.get("schema") != SCHEMA or not isinstance(doc.get("tasks"), dict):
            raise ValueError("JSON 必须使用 medcl.predictions.v1 格式")
    else:
        raise ValueError("预测仅接受 JSON 或 NPZ/ZIP 数组包")


def required_tasks(config: dict, stage: int) -> list[str]:
    return config["order"] if config["evaluate_unseen"] else config["order"][:stage]


def load_predictions(path: Path) -> dict[str, dict]:
    data = path.read_bytes()
    inspect_upload(path.name, data, "predictions")
    if path.suffix == ".json":
        doc = parse_json(data)
        result = {}
        if len(doc["tasks"]) > 12:
            raise ValueError("预测任务数超过上限")
        for key, value in doc["tasks"].items():
            if not isinstance(value, dict) or set(value) != {"sample_ids", "predictions"}:
                raise ValueError("每个任务必须且仅包含 sample_ids 和 predictions")
            result[key] = {"ids": np.asarray(value["sample_ids"]), "pred": np.asarray(value["predictions"])}
        return result
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        task_ids = {key.split("__")[0] for key in archive.files}
        if set(archive.files) != {f"{t}__{field}" for t in task_ids for field in ("ids", "pred")}:
            raise ValueError("每个任务需同时提供 ids 和 pred 数组")
        return {t: {"ids": np.array(archive[f"{t}__ids"]), "pred": np.array(archive[f"{t}__pred"])} for t in task_ids}


def align_predictions(entry: dict, sample_ids: np.ndarray, expected_shape: tuple,
                      allowed_classes: list[int], kind: str) -> np.ndarray:
    ids, pred = entry["ids"], entry["pred"]
    if ids.ndim != 1 or ids.dtype.kind not in "US" or len(ids) != len(sample_ids) or len(set(ids.tolist())) != len(ids):
        raise ValueError("样本 ID 缺失、重复或格式错误")
    if ids.dtype.kind == "S":
        ids = ids.astype("U")
    if set(ids.tolist()) != set(sample_ids.tolist()):
        raise ValueError("预测与冻结测试集的样本 ID 不匹配")
    if pred.shape != expected_shape or pred.dtype.kind not in "biuf" or not np.isfinite(pred).all():
        raise ValueError("预测维度、数值类型或有限值检查失败")
    if kind != "registration" and (pred.dtype.kind not in "biu" or not np.isin(pred, allowed_classes).all()):
        raise ValueError("预测类别超出当前阶段输出空间，或不是整数标签")
    positions = {str(value): index for index, value in enumerate(ids)}
    return pred[[positions[str(sample)] for sample in sample_ids]]


def submit(benchmark: dict, *, method: str, order: list[str], uploads: list[dict],
           mode: str, architecture: str | None, clients: int, evaluate_unseen: bool,
           output_head: str = "shared", root: Path | None = None) -> str:
    ok, _ = readiness(benchmark)
    if not ok:
        raise ValueError("该基准资产尚未接入")
    if not isinstance(method, str) or not 1 <= len(method.strip()) <= 80 or any(ord(c) < 32 for c in method):
        raise ValueError("方法名称需为 1–80 个可见字符")
    task_ids = [t["id"] for t in benchmark["tasks"]]
    if len(order) != len(task_ids) or set(order) != set(task_ids):
        raise ValueError("任务顺序必须包含全部任务 ID，且不能重复")
    if mode not in ("model", "predictions") or type(clients) is not int or clients not in (1, 2, 3, 4):
        raise ValueError("提交模式或客户端数量无效")
    if output_head not in ("shared", "task-specific"):
        raise ValueError("输出头条件无效")
    if type(evaluate_unseen) is not bool:
        raise ValueError("未见任务选项必须为布尔值")
    if evaluate_unseen and (not benchmark["allow_unseen"] or output_head != "shared"):
        raise ValueError("该协议或输出头不允许评测未见任务")
    if not 1 <= len(uploads) <= len(order):
        raise ValueError("每个阶段最多一个文件；至少提交一个阶段")
    stages = [u["stage"] for u in uploads]
    if len(set(stages)) != len(stages) or any(type(s) is not int or not 1 <= s <= len(order) for s in stages):
        raise ValueError("阶段位置无效或重复")
    if sum(len(u["data"]) for u in uploads) > MAX_TOTAL:
        raise ValueError("全部上传合计不可超过 256 MiB")
    if mode == "model":
        from medcl.sandbox import sandbox_available
        if architecture not in ARCHITECTURES[benchmark["kind"]] or output_head != "shared":
            raise ValueError("所选模型结构或输出头尚未审核")
        if not sandbox_available():
            raise ValueError("本机尚未通过模型隔离检查，请改为上传预测")
    for item in uploads:
        inspect_upload(item["name"], item["data"], mode, architecture)
    assets = freeze_assets(benchmark)
    by_path = {a["path"]: a for a in assets}
    from medcl.benchmarks import asset_paths
    config = {
        "schema_version": 1, "benchmark": public_protocol(benchmark), "method": method.strip(),
        "order": list(order), "stages": sorted(stages), "mode": mode, "architecture": architecture,
        "clients": clients, "evaluate_unseen": bool(evaluate_unseen), "output_head": output_head,
        "client_split": {"id": f"case-round-robin-v1-c{clients}", "version": "1",
                         "source": "平台固定逻辑划分：各任务匿名病例索引 mod 客户端数；分类无病例标识时按图像索引。非论文客户端划分。"},
        "conditions": {"segmentation": "病例级前景类 Dice (eps=1e-5)，另列含背景宏均值；同空=1",
                       "classification": "样本准确率；固定全局类别编码；仅已见类别输出；不推断病例 ID",
                       "registration": "固定空间对应点 TRE；有序坐标乘协议 spacing 后求欧氏距离；mm"}[benchmark["kind"]],
        "test_assets": [{"task_id": task["id"], "files": [{"size": by_path[str(p)]["size"], "mtime_ns": by_path[str(p)]["mtime_ns"]} for p in asset_paths(task)]} for task in benchmark["tasks"]],
        "prediction_provenance": "提交者声明同一阶段全局模型；预测模式不独立验证模型来源" if mode == "predictions" else "同一阶段上传权重用于所有逻辑客户端",
    }
    return create_job(config, {"benchmark": benchmark, "assets": assets}, uploads, root)
