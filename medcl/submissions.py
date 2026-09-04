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

from medcl import EVALUATOR_VERSION, VIEWER_SCHEMA_VERSION
from medcl.benchmarks import allowed_output_heads, freeze_assets, public_protocol, readiness
from medcl.storage import create_job

MAX_FILE = 128 * 1024 * 1024
MAX_JSON = 16 * 1024 * 1024
MAX_TOTAL = 256 * 1024 * 1024
MAX_EXPANDED = 512 * 1024 * 1024
MAX_JSON_PREDICTIONS = 2_000_000
MAX_JSON_IDS = 1_000_000
MAX_JSON_STRING_CHARS = 8_000_000
SCHEMA = "medcl.predictions.v1"
PROVENANCE = {
    "synthetic": "由已校验的合成协议决定；仅用于工程验收",
    "untrained_baseline": "提交者声明为未训练基线；平台未验证训练过程",
    "trained_model_declared": "提交者声明为已训练模型；平台未验证训练过程",
    "external_predictions_unknown": "外部预测来源未知；平台仅验证测试评分",
}
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


def validate_prediction_json(doc: object) -> dict:
    if not isinstance(doc, dict) or set(doc) != {"schema", "tasks"} or doc.get("schema") != SCHEMA or not isinstance(doc.get("tasks"), dict):
        raise ValueError("JSON 必须使用 medcl.predictions.v1 格式")
    tasks = doc["tasks"]
    if not 1 <= len(tasks) <= 12:
        raise ValueError("预测 JSON 必须包含 1–12 个任务")
    string_chars = 0
    prediction_count = 0
    for task_id, value in tasks.items():
        if not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,63}", task_id):
            raise ValueError("预测任务 ID 格式错误")
        if not isinstance(value, dict) or set(value) != {"sample_ids", "predictions"}:
            raise ValueError("每个任务必须且仅包含 sample_ids 和 predictions")
        ids = value["sample_ids"]
        if not isinstance(ids, list) or not 1 <= len(ids) <= MAX_JSON_IDS:
            raise ValueError("sample_ids 数量超出上限")
        if any(not isinstance(item, str) or not 1 <= len(item) <= 128 or any(ord(c) < 32 for c in item) for item in ids):
            raise ValueError("sample_ids 必须是有界可见字符串")
        string_chars += sum(map(len, ids))
        if string_chars > MAX_JSON_STRING_CHARS:
            raise ValueError("JSON 内容超过资源上限；请改用 NPZ")
        stack = [(value["predictions"], 0)]
        while stack:
            item, depth = stack.pop()
            if depth > 5:
                raise ValueError("predictions 嵌套过深；大型预测请使用 NPZ")
            if isinstance(item, list):
                if not item:
                    raise ValueError("predictions 不得包含空数组")
                stack.extend((child, depth + 1) for child in item)
            elif isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError("predictions 只能包含有限数值")
            else:
                try:
                    finite = math.isfinite(item)
                except OverflowError:
                    finite = False
                if not finite:
                    raise ValueError("predictions 只能包含有限数值")
                prediction_count += 1
                if prediction_count > MAX_JSON_PREDICTIONS:
                    raise ValueError("JSON 内容超过资源上限；请改用 NPZ")
    return doc


def validate_archive(data: bytes, *, allow_registration_volumes: bool = False) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if not 2 <= len(entries) <= 48 or len({e.filename for e in entries}) != len(entries):
                raise ValueError("NPZ 必须有 1–12 个任务的成对数组，且不可重复")
            if sum(e.file_size for e in entries) > MAX_EXPANDED:
                raise ValueError("NPZ 解压后超过 512 MiB 上限")
            for entry in entries:
                path = PurePosixPath(entry.filename)
                match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9-]{0,63})__(ids|pred|registered|warped_prediction)\.npy", entry.filename)
                if (path.is_absolute() or len(path.parts) != 1 or match is None
                        or (match.group(2) in ("registered", "warped_prediction") and not allow_registration_volumes)):
                    raise ValueError("ZIP 路径非法或包含当前协议不允许的数组")
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
                    field = match.group(2)
                    if (dtype.hasobject or dtype.kind not in "biufUS" or len(shape) > 5
                            or (field == "ids" and (dtype.kind not in "US" or len(shape) != 1))
                            or (field in ("pred", "registered") and dtype.kind not in "iuf")
                            or (field == "warped_prediction" and (dtype.kind not in "iu" or len(shape) != 4))
                            or (field == "registered" and len(shape) != 4)):
                        raise ValueError("不接受 Pickle、对象或结构化数组")
                    if not shape or any(d <= 0 for d in shape) or math.prod(shape) * dtype.itemsize > MAX_EXPANDED:
                        raise ValueError("数组维度为空或超过资源上限")
                    if handle.tell() + math.prod(shape) * dtype.itemsize != entry.file_size:
                        raise ValueError("数组声明大小与文件内容不符")
            names = {entry.filename[:-4] for entry in entries}
            tasks = {name.split("__", 1)[0] for name in names}
            if not 1 <= len(tasks) <= 12 or any({f"{task}__ids", f"{task}__pred"} - names for task in tasks):
                raise ValueError("每个任务需同时提供 ids 和 pred 数组")
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


def inspect_upload(name: str, data: bytes, mode: str, architecture: str | None = None,
                   *, allow_registration_volumes: bool = False):
    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_FILE:
        raise ValueError("每个文件需为 1 byte–128 MiB")
    suffix = Path(name).suffix.lower()
    if mode == "model":
        if suffix != ".safetensors":
            raise ValueError("模型仅接受 safetensors；禁止 Python、Pickle、PT/PTH")
        if architecture not in {item for choices in ARCHITECTURES.values() for item in choices}:
            raise ValueError("模型结构未在审核白名单中")
        validate_weights(data, architecture)
        return None
    if mode != "predictions":
        raise ValueError("提交模式无效")
    if suffix in (".npz", ".zip"):
        validate_archive(data, allow_registration_volumes=allow_registration_volumes)
        return None
    if suffix == ".json":
        if len(data) > MAX_JSON:
            raise ValueError("JSON 不可超过 16 MiB；大型预测请使用 NPZ")
        return validate_prediction_json(parse_json(data))
    else:
        raise ValueError("预测仅接受 JSON 或 NPZ/ZIP 数组包")


def required_tasks(config: dict, stage: int) -> list[str]:
    return config["order"] if config["evaluate_unseen"] else config["order"][:stage]


def load_predictions(path: Path) -> dict[str, dict]:
    data = path.read_bytes()
    doc = inspect_upload(path.name, data, "predictions", allow_registration_volumes=True)
    if path.suffix == ".json":
        result = {}
        for key, value in doc["tasks"].items():
            result[key] = {"ids": np.asarray(value["sample_ids"]), "pred": np.asarray(value["predictions"])}
        return result
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        task_ids = {key.split("__", 1)[0] for key in archive.files}
        result = {}
        for task in task_ids:
            fields = {key.split("__", 1)[1] for key in archive.files if key.startswith(f"{task}__")}
            if not {"ids", "pred"} <= fields or not fields <= {"ids", "pred", "registered", "warped_prediction"}:
                raise ValueError("每个任务需同时提供 ids 和 pred 数组")
            result[task] = {field: np.array(archive[f"{task}__{field}"]) for field in fields}
        return result


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


def align_registration_volumes(entry: dict, sample_ids: np.ndarray, volume_shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    """Align optional qualitative registration volumes with the already checked IDs."""
    ids = entry["ids"].astype("U") if entry["ids"].dtype.kind == "S" else entry["ids"]
    positions = {str(value): index for index, value in enumerate(ids)}
    order = [positions[str(sample)] for sample in sample_ids]
    result = {}
    for name in ("registered", "warped_prediction"):
        if name not in entry:
            continue
        array = entry[name]
        if array.shape != volume_shape:
            raise ValueError("配准后体数据必须与 fixed display grid 完全同形")
        if name == "registered":
            if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
                raise ValueError("配准后体数据必须是有限数值 scalar volume")
        elif (array.dtype.kind not in "iu" or np.any(array < 0) or array.max(initial=0) > 65535):
            raise ValueError("warped prediction 必须是 uint16 范围内的整数 labelmap")
        result[name] = array[order]
    return result


def submit(benchmark: dict, *, method: str, order: list[str], uploads: list[dict],
           mode: str, architecture: str | None, clients: int, evaluate_unseen: bool,
           output_head: str = "shared", training_supervision: str | None = None,
           provenance: str | None = None, root: Path | None = None) -> str:
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
    if output_head not in allowed_output_heads(benchmark):
        raise ValueError("输出头不在该协议允许范围内")
    supervision = training_supervision or ("not-declared" if benchmark["kind"] == "segmentation" else "not-applicable")
    if benchmark["kind"] == "segmentation":
        if supervision not in ("full", "weak", "not-declared"):
            raise ValueError("分割训练监督类型无效")
    elif supervision != "not-applicable":
        raise ValueError("当前任务不使用分割训练监督类型")
    if type(evaluate_unseen) is not bool:
        raise ValueError("未见任务选项必须为布尔值")
    if evaluate_unseen and (not benchmark["allow_unseen"] or output_head != "shared"):
        raise ValueError("该协议或输出头不允许评测未见任务")
    if benchmark["synthetic"]:
        if provenance not in (None, "synthetic"):
            raise ValueError("合成协议的结果来源必须标记为 synthetic")
        provenance = "synthetic"
    else:
        provenance = provenance or "external_predictions_unknown"
        if provenance not in PROVENANCE or provenance == "synthetic":
            raise ValueError("真实协议结果来源必须使用受限声明类别")
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
        if any(task["format"] == "registration-volume" for task in benchmark["tasks"]):
            raise ValueError("registration-volume 第一版只接受预测文件")
    allow_volumes = any(task["format"] == "registration-volume" for task in benchmark["tasks"])
    for item in uploads:
        inspect_upload(item["name"], item["data"], mode, architecture,
                       allow_registration_volumes=allow_volumes)
    assets = freeze_assets(benchmark)
    by_path = {a["path"]: a for a in assets}
    from medcl.benchmarks import asset_paths
    config = {
        "schema_version": 2, "evaluator_version": EVALUATOR_VERSION,
        "viewer_schema_version": VIEWER_SCHEMA_VERSION,
        "benchmark": public_protocol(benchmark), "method": method.strip(),
        "order": list(order), "stages": sorted(stages), "mode": mode, "architecture": architecture,
        "clients": clients, "evaluate_unseen": bool(evaluate_unseen), "output_head": output_head,
        "training_supervision": supervision,
        "supervision_source": "提交者声明的外部训练监督类型；平台仅在同一冻结测试集评分，不读取训练集或训练日志" if benchmark["kind"] == "segmentation" else "不适用",
        "client_split": {"id": f"case-round-robin-v1-c{clients}", "version": "1",
                         "source": "平台固定逻辑划分：各任务匿名病例索引 mod 客户端数；分类无病例标识时按图像索引。非论文客户端划分。"},
        "conditions": {"segmentation": "病例级前景类 Dice (eps=1e-5)，另列含背景宏均值；同空=1",
                       "classification": "任务级样本准确率；固定全局类别编码；仅已见类别输出；不发布逐样本正误",
                       "registration": "固定空间对应点 TRE；有序坐标乘协议 spacing 后求欧氏距离；mm"}[benchmark["kind"]],
        "test_assets": [{"task_id": task["id"], "files": [{"size": by_path[str(p)]["size"], "mtime_ns": by_path[str(p)]["mtime_ns"]} for p in asset_paths(task)]} for task in benchmark["tasks"]],
        "provenance": {"category": provenance, "statement": PROVENANCE[provenance],
                       "training_verified": False},
    }
    return create_job(config, {"benchmark": benchmark, "assets": assets}, uploads, root)
