"""Operator-owned protocols. No browser input is interpreted as a local path."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re

import h5py
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
KINDS = {"classification": "分类", "segmentation": "分割", "registration": "配准"}
INCREMENTS = {"domain": "域增量", "class": "类别增量", "task": "任务增量"}
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,63}")
_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
_IDENTITY_DIRECTION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
_METRICS = {
    "classification": ("Accuracy", "higher", "fraction"),
    "segmentation": ("Foreground Dice", "higher", "fraction"),
    "registration": ("TRE", "lower", "mm"),
}
_FORMATS = {
    "classification": {"synthetic", "npy", "medmnist"},
    "segmentation": {"synthetic", "h5"},
    "registration": {"synthetic", "landmarks", "registration-volume"},
}
_PATH_FIELDS = {
    "npy": ("images_path", "labels_path"),
    "medmnist": ("path",),
    "h5": ("path",),
    "landmarks": ("path",),
    "registration-volume": ("path",),
}


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("管理员配置包含重复键")
        result[key] = value
    return result


def _text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 4096 or any(ord(c) < 32 for c in value):
        raise ValueError(f"{name} 必须是非空可见文本")
    return value


def _id(value, name: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ValueError(f"{name} 只允许 1–64 位 ASCII 字母、数字和短横线")
    return value


def _integer_list(value, name: str, *, nonempty: bool = False) -> list[int]:
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{name} 必须是{'非空' if nonempty else ''}整数列表")
    if any(type(item) is not int for item in value) or len(value) != len(set(value)):
        raise ValueError(f"{name} 必须由不重复整数构成")
    return value


def _spacing(value, name: str, length: int | None = None) -> list[float]:
    if not isinstance(value, list) or not value or (length is not None and len(value) != length):
        raise ValueError(f"{name} 必须是{'恰好 ' + str(length) + ' 个' if length else '非空'}正数列表")
    for item in value:
        try:
            valid = not isinstance(item, bool) and isinstance(item, (int, float)) and math.isfinite(item) and item > 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f"{name} 必须是有限正数")
    return value


def _finite_vector(value, name: str, length: int) -> list[float]:
    try:
        valid = (isinstance(value, list) and len(value) == length and all(
            not isinstance(item, bool) and isinstance(item, (int, float)) and math.isfinite(item) for item in value))
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} 必须是恰好 {length} 个有限数值")
    return value


def allowed_output_heads(benchmark: dict) -> list[str]:
    """Return the validated protocol head choices, including legacy defaults."""
    configured = benchmark.get("allowed_output_heads")
    if configured is not None:
        if (not isinstance(configured, list) or not configured or len(configured) != len(set(configured))
                or any(head not in ("shared", "task-specific") for head in configured)):
            raise ValueError("allowed_output_heads 只能由 shared / task-specific 构成")
        return configured
    if benchmark.get("kind") == "segmentation" and benchmark.get("incremental") == "task":
        spaces = [(task.get("classes"), task.get("all_classes")) for task in benchmark.get("tasks", [])]
        if spaces and any(space != spaces[0] for space in spaces[1:]):
            return ["task-specific"]
    return ["shared", "task-specific"] if benchmark.get("kind") == "segmentation" else ["shared"]


def validate_task(benchmark: dict, task: dict) -> None:
    """Validate one operator-owned task without opening its private assets."""
    if not isinstance(task, dict):
        raise ValueError("管理员任务必须是对象")
    _id(task.get("id"), "任务 ID")
    _text(task.get("name"), "任务名称")
    fmt = task.get("format")
    kind = benchmark["kind"]
    if fmt not in _FORMATS[kind]:
        raise ValueError("任务资产格式与大任务类型不匹配")
    generated_volume = benchmark["synthetic"] and kind == "registration" and fmt == "registration-volume"
    if (fmt == "synthetic" and not benchmark["synthetic"]) or (benchmark["synthetic"] and fmt != "synthetic" and not generated_volume):
        raise ValueError("synthetic 标志与任务资产格式不一致")
    for key in _PATH_FIELDS.get(fmt, ()):
        value = _text(task.get(key), f"任务 {task['id']} 的 {key}")
        if not Path(value).expanduser().is_absolute():
            raise ValueError("真实测试资产必须使用管理员配置的绝对路径")

    if kind in ("classification", "segmentation"):
        classes = _integer_list(task.get("classes"), "classes", nonempty=True)
        all_classes = _integer_list(task.get("all_classes"), "all_classes", nonempty=True)
        if any(label < 0 for label in all_classes):
            raise ValueError("类别 ID 必须是非负整数")
        if not set(classes) <= set(all_classes):
            raise ValueError("classes 必须是 all_classes 的子集")
        if kind == "segmentation":
            if 0 not in all_classes or 0 in classes:
                raise ValueError("分割 all_classes 必须含背景 0，主指标 classes 不得含背景")
            shift = task.get("label_shift", 0)
            if type(shift) is not int or shift < 0:
                raise ValueError("label_shift 必须是非负整数")
            if any(label > 65535 for label in all_classes):
                raise ValueError("分割类别 ID 必须在 uint16 可视化范围内")
            if "voxel_spacing_zyx" in task:
                _spacing(task["voxel_spacing_zyx"], "voxel_spacing_zyx", 3)
    elif any(key in task for key in ("classes", "all_classes", "label_shift")):
        raise ValueError("配准任务不得使用分类或分割类别字段")

    if kind == "registration":
        coordinate_system = _text(task.get("coordinate_system"), "coordinate_system")
        if fmt == "synthetic":
            spacing = _spacing(task.get("spacing"), "spacing", 2)
        elif fmt == "landmarks":
            spacing = _spacing(task.get("spacing"), "spacing", 3)
        else:
            spacing = task.get("spacing", [1, 1, 1])
            if spacing != [1, 1, 1]:
                raise ValueError("registration-volume 点坐标已为 mm，spacing 必须省略或为 [1,1,1]")
            _spacing(task.get("voxel_spacing_zyx", [1, 1, 1]), "voxel_spacing_zyx", 3)
            if "origin_xyz" in task:
                _finite_vector(task["origin_xyz"], "origin_xyz", 3)
            if "direction_xyz" in task:
                direction = _finite_vector(task["direction_xyz"], "direction_xyz", 9)
                if any(abs(value - expected) > 1e-6 for value, expected in zip(direction, _IDENTITY_DIRECTION)):
                    raise ValueError("当前 registration-volume 只支持 identity direction")
        if fmt == "synthetic" and (coordinate_system != "fixed-space xy, mm" or spacing != [1, 1]):
            raise ValueError("合成配准协议固定使用 fixed-space xy, mm 且 spacing=[1,1]")
        if fmt == "landmarks" and (coordinate_system != "fixed-space xyz, mm" or spacing != [1, 1, 1]):
            raise ValueError("真实配准 v1 仅接受 fixed-space xyz, mm 且 spacing=[1,1,1]")
        if fmt == "registration-volume" and coordinate_system != "fixed-display-grid xyz, mm":
            raise ValueError("registration-volume 只接受 fixed-display-grid xyz, mm")


def validate_benchmark(benchmark: dict) -> dict:
    """Central fail-closed schema for configured, pending, and demo protocols."""
    if not isinstance(benchmark, dict):
        raise ValueError("管理员基准必须是对象")
    _id(benchmark.get("id"), "基准 ID")
    for key, label in (("title", "基准标题"),
                       ("description", "协议说明"), ("source", "协议来源"),
                       ("output_semantics", "输出语义")):
        _text(benchmark.get(key), label)
    if not isinstance(benchmark.get("version"), str) or _VERSION.fullmatch(benchmark["version"]) is None:
        raise ValueError("协议版本必须是稳定 ASCII 标识")
    kind, incremental = benchmark.get("kind"), benchmark.get("incremental")
    if kind not in KINDS or incremental not in INCREMENTS:
        raise ValueError("管理员基准类型或增量场景无效")
    if kind == "classification" and incremental != "class":
        raise ValueError("当前分类平台只登记类别增量协议")
    if kind == "registration" and incremental != "task":
        raise ValueError("当前配准平台只登记任务增量协议")
    if type(benchmark.get("synthetic")) is not bool or type(benchmark.get("allow_unseen")) is not bool:
        raise ValueError("synthetic 和 allow_unseen 必须是严格布尔值")
    if benchmark.get("pending", False) not in (True, False) or type(benchmark.get("pending", False)) is not bool:
        raise ValueError("pending 必须是严格布尔值")
    if (benchmark.get("metric"), benchmark.get("direction"), benchmark.get("unit")) != _METRICS[kind]:
        raise ValueError("metric、direction、unit 与大任务类型不匹配")
    tasks = benchmark.get("tasks")
    if not isinstance(tasks, list) or (not benchmark.get("pending") and not 1 <= len(tasks) <= 12) or len(tasks) > 12:
        raise ValueError("非待接入协议任务数必须为 1–12")
    for task in tasks:
        validate_task(benchmark, task)
    task_ids = [task["id"] for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("管理员任务 ID 必须唯一")
    formats = {task["format"] for task in tasks}
    generated_registration = kind == "registration" and formats == {"registration-volume"}
    if benchmark["synthetic"] and formats != {"synthetic"} and not generated_registration:
        raise ValueError("合成协议不得混入真实资产任务")
    if not benchmark["synthetic"] and "synthetic" in formats:
        raise ValueError("真实协议不得包含合成任务")
    heads = allowed_output_heads(benchmark)
    if benchmark.get("pending"):
        return benchmark

    if kind in ("classification", "segmentation") and incremental in ("domain", "class"):
        all_classes = tasks[0]["all_classes"]
        if any(task["all_classes"] != all_classes for task in tasks[1:]):
            raise ValueError("同一协议各任务必须登记一致的 all_classes 顺序")
    if incremental == "domain":
        if any(task["classes"] != tasks[0]["classes"] for task in tasks[1:]):
            raise ValueError("域增量任务必须登记一致的 classes 顺序")
    if kind == "classification":
        names = benchmark.get("class_names")
        if not isinstance(names, dict) or set(names) != {str(label) for label in all_classes}:
            raise ValueError("class_names 必须完整覆盖全局类别 ID")
        if any(not isinstance(name, str) or not name.strip() or len(name) > 160 or any(ord(c) < 32 for c in name) for name in names.values()):
            raise ValueError("class_names 名称必须是非空文本")
    if incremental == "class":
        seen = set()
        for task in tasks:
            current = set(task["classes"])
            if seen & current:
                raise ValueError("类别增量任务的 classes 不得重叠")
            seen |= current
        expected = set(all_classes) - ({0} if kind == "segmentation" else set())
        if seen != expected:
            raise ValueError("类别增量任务集合必须完整覆盖全局前景类别")
    if kind == "segmentation" and incremental == "task":
        spaces = [(task["classes"], task["all_classes"]) for task in tasks]
        if any(space != spaces[0] for space in spaces[1:]) and "shared" in heads:
            raise ValueError("标签空间不一致的任务增量分割只允许 task-specific 输出头")
    return benchmark


def config_path() -> Path:
    return Path(os.environ.get("MEDCL_CONFIG", PROJECT / "settings.local.json")).expanduser()


def state_path() -> Path:
    return Path(os.environ.get("MEDCL_STATE_DIR", Path.home() / ".local/state/medcl")).expanduser().resolve()


def demo_protocol(kind: str) -> dict:
    protocol = {
        "id": f"demo-{kind}", "title": f"{KINDS[kind]} · 合成验收样例",
        "kind": kind, "incremental": {"classification": "class", "segmentation": "domain", "registration": "task"}[kind],
        "version": "synthetic-v1", "synthetic": True,
        "description": "程序生成的手算 / 工程验收数据；不代表医学数据、训练效果或论文结果。",
        "source": "MedCL deterministic synthetic fixture v1",
        "metric": {"classification": "Accuracy", "segmentation": "Foreground Dice", "registration": "TRE"}[kind],
        "direction": "lower" if kind == "registration" else "higher",
        "unit": "mm" if kind == "registration" else "fraction",
        "allow_unseen": kind == "segmentation",
        "allowed_output_heads": ["shared", "task-specific"] if kind == "segmentation" else ["shared"],
        "output_semantics": "shared-global-labels" if kind != "registration" else "fixed-space-corresponding-landmarks",
        "tasks": [{"id": f"T{i + 1}", "name": f"合成任务 {i + 1}", "format": "synthetic",
                   **({"classes": [1], "all_classes": [0, 1]} if kind == "segmentation" else
                      {"classes": [i], "all_classes": [0, 1, 2]} if kind == "classification" else
                      {"spacing": [1.0, 1.0], "coordinate_system": "fixed-space xy, mm"})}
                  for i in range(3)],
    }
    if kind == "classification":
        protocol["class_names"] = {str(i): f"合成类别 {i}" for i in range(3)}
    return protocol


def pending_protocols() -> list[dict]:
    entries = [
        ("prostate-domain6", "前列腺分割 · 六中心", "segmentation", "domain", "病例级前景 Dice；需管理员接入冻结测试 HDF5。"),
        ("mmwhs-class3", "心脏分割 · 三阶段类别增量", "segmentation", "class", "MYO/LV/LA → RA/RV → AO/PA；需确认全局类别映射及输出头。"),
        ("organ-task4", "器官分割 · 四任务", "segmentation", "task", "左心房、前列腺、肝脏、脑肿瘤；需接入各任务测试集。"),
        ("pathmnist-class3", "PathMNIST · 三阶段类别增量", "classification", "class", "平台自定义类别分组，不冒充 FedSubMerge 原论文协议。"),
        ("registration-landmarks", "配准 · 对应标志点", "registration", "task", "需接入版本化图像对、固定空间标志点、坐标与 mm 单位约定；不冒充 SAMCL 结果。"),
    ]
    return [{"id": bid, "title": title, "kind": kind, "incremental": inc,
             "version": "not-connected", "synthetic": False, "description": desc,
             "source": "待管理员确认", "tasks": [], "pending": True,
             "metric": "TRE" if kind == "registration" else "Accuracy" if kind == "classification" else "Foreground Dice",
             "direction": "lower" if kind == "registration" else "higher", "unit": "mm" if kind == "registration" else "fraction",
            "allow_unseen": False, "output_semantics": "pending", "allowed_output_heads": ["shared"]}
            for bid, title, kind, inc, desc in entries]


def catalog() -> list[dict]:
    configured = []
    path = config_path()
    if path.is_file():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs)
        except (UnicodeError, json.JSONDecodeError):
            raise ValueError("管理员基准配置不是有效 JSON") from None
        if not isinstance(doc, dict):
            raise ValueError("管理员基准配置必须是对象")
        configured = doc.get("benchmarks", [])
        if not isinstance(configured, list):
            raise ValueError("管理员基准配置格式错误")
        for b in configured:
            validate_benchmark(b)
    configured_ids = {b["id"] for b in configured}
    if len(configured_ids) != len(configured):
        raise ValueError("管理员基准 ID 重复")
    result = configured + [b for b in pending_protocols() if b["id"] not in configured_ids] + [demo_protocol(k) for k in KINDS]
    for benchmark in result:
        validate_benchmark(benchmark)
    return result


def asset_paths(task: dict) -> list[Path]:
    return [Path(task[key]).expanduser().resolve() for key in ("path", "images_path", "labels_path") if task.get(key)]


def readiness(benchmark: dict) -> tuple[bool, str]:
    validate_benchmark(benchmark)
    if benchmark.get("pending") or not benchmark.get("tasks"):
        return False, "待接入测试数据与协议"
    missing = [t["id"] for t in benchmark["tasks"] if any(not p.is_file() for p in asset_paths(t))]
    if missing:
        return False, "待接入任务：" + "、".join(missing)
    return True, "合成验收数据" if benchmark.get("synthetic") else "已配置本地测试资产"


def public_protocol(benchmark: dict) -> dict:
    """Explicit allow-list prevents file paths/hidden labels escaping in reports."""
    keys = ("id", "title", "public_title", "kind", "incremental", "version", "synthetic", "description", "source", "class_names",
            "metric", "direction", "unit", "allow_unseen", "output_semantics", "preprocessing", "label_rule")
    out = {k: benchmark[k] for k in keys if k in benchmark}
    out["allowed_output_heads"] = allowed_output_heads(benchmark)
    task_keys = ("id", "name", "classes", "all_classes", "coordinate_system", "spacing", "voxel_spacing_zyx",
                 "origin_xyz", "direction_xyz", "source", "label_shift")
    out["tasks"] = [{k: t[k] for k in task_keys if k in t} for t in benchmark["tasks"]]
    return out


def freeze_assets(benchmark: dict) -> list[dict]:
    validate_benchmark(benchmark)
    paths = sorted({str(p) for t in benchmark["tasks"] for p in asset_paths(t)})
    if not benchmark["synthetic"] and not paths:
        raise ValueError("真实协议没有可冻结测试资产")
    records = []
    for path in paths:
        try:
            stat = Path(path).stat()
        except OSError:
            raise ValueError("测试资产离线；请管理员重新挂载后新建评测") from None
        if not Path(path).is_file():
            raise ValueError("测试资产不是普通文件")
        records.append({"path": path, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return records


def check_assets(records: list[dict]) -> None:
    for record in records:
        try:
            stat = Path(record["path"]).stat()
        except OSError:
            raise ValueError("测试资产离线；请管理员重新挂载后新建评测") from None
        if (stat.st_size, stat.st_mtime_ns) != (record["size"], record["mtime_ns"]):
            raise ValueError("测试资产在提交后发生变化；必须确认版本并新建评测")


def _synthetic(kind: str, index: int) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    rng = np.random.default_rng(101 + index)
    if kind == "classification":
        x = rng.normal(size=(12 + index * 3, 2)).astype(np.float32)
        # Disjoint class tasks with global output labels; no target-derived inference.
        x[:, 0] += (2, -2, -2)[index]
        x[:, 1] += (-2, 2, -2)[index]
        y = np.full(len(x), index, dtype=np.int64)
        return x, y, [(i, i + 1) for i in range(len(x))]
    if kind == "segmentation":
        y = np.zeros((6, 12, 12), dtype=np.int64)
        y[:, 3:8, 4:9] = 1
        y[0] = 0
        x = (y * 0.8 + rng.normal(0.1, 0.15, y.shape)).astype(np.float32)
        return x, y, [(0, 2), (2, 4), (4, 6)]
    target = rng.normal(size=(5, 4, 2)).astype(np.float32) * 5
    return target + np.array([1.0, -2.0], dtype=np.float32), target, [(i, i + 1) for i in range(5)]


def _ranges(ends, length: int) -> list[tuple[int, int]]:
    ends = np.asarray(ends)
    if ends.ndim != 1 or not len(ends) or not np.isfinite(ends).all() or not np.equal(ends, np.floor(ends)).all():
        raise ValueError("病例边界必须为一维整数")
    ends = ends.astype(np.int64)
    if ends[0] < 0 or ends[-1] != length - 1 or np.any(np.diff(ends) <= 0):
        raise ValueError("病例边界未完整覆盖测试样本")
    ranges = list(zip([0] + (ends[:-1] + 1).tolist(), (ends + 1).tolist()))
    if any(start >= end for start, end in ranges):
        raise ValueError("病例边界包含空病例")
    return ranges


def read_task(benchmark: dict, task: dict, *, include_targets: bool = True) -> dict:
    """Only fixed test splits. Classification has no inferred patient identities."""
    kind, fmt = benchmark["kind"], task["format"]
    if fmt == "synthetic":
        images, labels, ranges = _synthetic(kind, [t["id"] for t in benchmark["tasks"]].index(task["id"]))
        original_indices = np.arange(len(images))
    elif fmt == "h5":
        with h5py.File(Path(task["path"]).expanduser(), "r") as f:
            shape = f["test_images"].shape
            if len(shape) != 3 or f["test_labels"].shape != shape:
                raise ValueError("HDF5 测试图像和标签维度不兼容")
            images = np.moveaxis(np.asarray(f["test_images"], dtype=np.float32), -1, 0)
            labels = None
            if include_targets:
                raw = np.moveaxis(np.asarray(f["test_labels"]), -1, 0)
                if not np.isfinite(raw).all() or np.any(raw < 0):
                    raise ValueError("测试标签包含无效数值")
                # Match the reviewed DenseDataset: truncate interpolated labels, never threshold.
                labels = raw.astype(np.int64)
                shift = int(task.get("label_shift", 0))
                if shift:
                    labels = np.where(labels > 0, labels + shift, 0)
            ranges = _ranges(f["patient_info_test"][:], len(images))
            original_indices = np.arange(len(images))
    elif fmt in ("npy", "medmnist"):
        if fmt == "npy":
            images = np.load(task["images_path"], mmap_mode="r", allow_pickle=False)
            raw_labels = np.load(task["labels_path"], mmap_mode="r", allow_pickle=False)
        else:
            with np.load(task["path"], allow_pickle=False) as f:
                images, raw_labels = f["test_images"], f["test_labels"]
        raw_labels = np.asarray(raw_labels).reshape(-1)
        if len(raw_labels) != len(images) or not np.isfinite(raw_labels).all() or not np.equal(raw_labels, np.floor(raw_labels)).all():
            raise ValueError("分类测试标签或样本数量错误")
        original_indices = np.flatnonzero(np.isin(raw_labels, task["classes"])) if task.get("filter_classes", True) else np.arange(len(images))
        images = np.asarray(images[original_indices])
        labels = raw_labels[original_indices].astype(np.int64) if include_targets else None
        ranges = [(i, i + 1) for i in range(len(images))]
    elif fmt in ("landmarks", "registration-volume"):
        with np.load(task["path"], allow_pickle=False) as f:
            expected = {"moving_points", "fixed_points"} if fmt == "landmarks" else {
                "fixed_volumes", "moving_volumes", "moving_points", "fixed_points"}
            if set(f.files) != expected:
                raise ValueError("配准测试资产键不符合已登记格式")
            images = np.asarray(f["moving_points"], dtype=np.float64)
            labels = np.asarray(f["fixed_points"], dtype=np.float64) if include_targets else None
            if fmt == "registration-volume":
                fixed_volumes = np.asarray(f["fixed_volumes"])
                moving_volumes = np.asarray(f["moving_volumes"])
        if fmt == "landmarks" and (task.get("coordinate_system") != "fixed-space xyz, mm" or task.get("spacing") != [1, 1, 1]):
            raise ValueError("真实配准 v1 仅接受固定空间 xyz 毫米坐标")
        if (images.ndim != 3 or images.shape[-1] != 3
                or labels is not None and (labels.shape != images.shape or not np.isfinite(labels).all())):
            raise ValueError("配准点必须为同形 [N,K,3] 数组")
        if fmt == "registration-volume":
            if (fixed_volumes.ndim != 4 or fixed_volumes.shape != moving_volumes.shape
                    or fixed_volumes.shape[0] != images.shape[0] or 0 in fixed_volumes.shape
                    or not np.isfinite(fixed_volumes).all() or not np.isfinite(moving_volumes).all()):
                raise ValueError("配准体数据必须为同形有限 [N,Z,Y,X] 固定显示网格")
        ranges = [(i, i + 1) for i in range(len(images))]
        original_indices = np.arange(len(images))
    else:
        raise ValueError("尚未支持该测试资产格式")
    if not len(images) or not np.isfinite(images).all():
        raise ValueError("测试输入为空或包含非有限值")
    if include_targets and kind != "registration":
        allowed_labels = ({0} | set(task["classes"])) if kind == "segmentation" else set(task["all_classes"])
        if not np.isin(labels, list(allowed_labels)).all():
            raise ValueError("测试标签不符合已登记类别集合")
    result = {"images": images, "target": labels if include_targets else None, "ranges": ranges,
            "sample_ids": np.asarray([f"{task['id']}-s{int(i):06d}" for i in original_indices]),
            "case_ids": [f"{task['id']}-c{i:04d}" for i in range(len(ranges))]}
    if fmt == "registration-volume":
        result.update(fixed_volumes=fixed_volumes, moving_volumes=moving_volumes)
    return result
