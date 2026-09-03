"""Operator-owned protocols. No browser input is interpreted as a local path."""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
KINDS = {"classification": "分类", "segmentation": "分割", "registration": "配准"}
INCREMENTS = {"domain": "域增量", "class": "类别增量", "task": "任务增量"}


def config_path() -> Path:
    return Path(os.environ.get("MEDCL_CONFIG", PROJECT / "settings.local.json")).expanduser()


def state_path() -> Path:
    return Path(os.environ.get("MEDCL_STATE_DIR", Path.home() / ".local/state/medcl")).expanduser().resolve()


def demo_protocol(kind: str) -> dict:
    return {
        "id": f"demo-{kind}", "title": f"{KINDS[kind]} · 合成验收样例",
        "kind": kind, "incremental": {"classification": "class", "segmentation": "domain", "registration": "task"}[kind],
        "version": "synthetic-v1", "synthetic": True,
        "description": "程序生成的手算 / 工程验收数据；不代表医学数据、训练效果或论文结果。",
        "source": "MedCL deterministic synthetic fixture v1",
        "metric": {"classification": "Accuracy", "segmentation": "Foreground Dice", "registration": "TRE"}[kind],
        "direction": "lower" if kind == "registration" else "higher",
        "unit": "mm" if kind == "registration" else "fraction",
        "allow_unseen": kind == "segmentation",
        "output_semantics": "shared-global-labels" if kind != "registration" else "fixed-space-corresponding-landmarks",
        "tasks": [{"id": f"T{i + 1}", "name": f"合成任务 {i + 1}", "format": "synthetic",
                   "classes": [1] if kind == "segmentation" else [i] if kind == "classification" else [],
                   "all_classes": [0, 1] if kind == "segmentation" else [0, 1, 2],
                   "spacing": [1.0, 1.0] if kind == "registration" else None,
                   "coordinate_system": "fixed-space xy, mm" if kind == "registration" else None}
                  for i in range(3)],
    }


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
             "allow_unseen": False, "output_semantics": "pending"}
            for bid, title, kind, inc, desc in entries]


def catalog() -> list[dict]:
    configured = []
    path = config_path()
    if path.is_file():
        doc = json.loads(path.read_text(encoding="utf-8"))
        configured = doc.get("benchmarks", [])
        if not isinstance(configured, list):
            raise ValueError("管理员基准配置格式错误")
        for b in configured:
            if b["kind"] not in KINDS or b["incremental"] not in INCREMENTS or not b.get("version"):
                raise ValueError("管理员基准类型或版本无效")
            ids = [t["id"] for t in b["tasks"]]
            if not ids or len(ids) != len(set(ids)) or len(ids) > 12:
                raise ValueError("管理员任务 ID 必须唯一，任务数为 1–12")
            for task in b["tasks"]:
                if not task["id"].isascii() or not task["id"].replace("-", "").isalnum():
                    raise ValueError("任务 ID 只允许 ASCII 字母、数字和短横线")
            b.setdefault("synthetic", False)
            b.setdefault("allow_unseen", False)
            b.setdefault("unit", "mm" if b["kind"] == "registration" else "fraction")
            b.setdefault("direction", "lower" if b["kind"] == "registration" else "higher")
    configured_ids = {b["id"] for b in configured}
    if len(configured_ids) != len(configured):
        raise ValueError("管理员基准 ID 重复")
    return configured + [b for b in pending_protocols() if b["id"] not in configured_ids] + [demo_protocol(k) for k in KINDS]


def asset_paths(task: dict) -> list[Path]:
    return [Path(task[key]).expanduser().resolve() for key in ("path", "images_path", "labels_path") if task.get(key)]


def readiness(benchmark: dict) -> tuple[bool, str]:
    if benchmark.get("pending") or not benchmark.get("tasks"):
        return False, "待接入测试数据与协议"
    missing = [t["id"] for t in benchmark["tasks"] if any(not p.is_file() for p in asset_paths(t))]
    if missing:
        return False, "待接入任务：" + "、".join(missing)
    return True, "合成验收数据" if benchmark.get("synthetic") else "已配置本地测试资产"


def public_protocol(benchmark: dict) -> dict:
    """Explicit allow-list prevents file paths/hidden labels escaping in reports."""
    keys = ("id", "title", "kind", "incremental", "version", "synthetic", "description", "source",
            "metric", "direction", "unit", "allow_unseen", "output_semantics", "preprocessing", "label_rule")
    out = {k: benchmark[k] for k in keys if k in benchmark}
    task_keys = ("id", "name", "classes", "all_classes", "coordinate_system", "spacing", "source", "label_shift")
    out["tasks"] = [{k: t[k] for k in task_keys if k in t} for t in benchmark["tasks"]]
    return out


def freeze_assets(benchmark: dict) -> list[dict]:
    paths = sorted({str(p) for t in benchmark["tasks"] for p in asset_paths(t)})
    return [{"path": p, "size": Path(p).stat().st_size, "mtime_ns": Path(p).stat().st_mtime_ns} for p in paths]


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
    if ends[-1] != length - 1 or np.any(np.diff(ends) < 0) or ends[0] < -1:
        raise ValueError("病例边界未完整覆盖测试样本")
    return list(zip([0] + (ends[:-1] + 1).tolist(), (ends + 1).tolist()))


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
    elif fmt == "landmarks":
        with np.load(task["path"], allow_pickle=False) as f:
            images = np.asarray(f["moving_points"], dtype=np.float64)
            labels = np.asarray(f["fixed_points"], dtype=np.float64) if include_targets else None
        if task.get("coordinate_system") != "fixed-space xyz, mm" or task.get("spacing") != [1, 1, 1]:
            raise ValueError("真实配准 v1 仅接受固定空间 xyz 毫米坐标")
        ranges = [(i, i + 1) for i in range(len(images))]
        original_indices = np.arange(len(images))
    else:
        raise ValueError("尚未支持该测试资产格式")
    if not len(images) or not np.isfinite(images).all():
        raise ValueError("测试输入为空或包含非有限值")
    if include_targets and kind != "registration":
        if not np.isin(labels, task["all_classes"]).all():
            raise ValueError("测试标签不符合已登记类别集合")
    return {"images": images, "target": labels if include_targets else None, "ranges": ranges,
            "sample_ids": np.asarray([f"{task['id']}-s{int(i):06d}" for i in original_indices]),
            "case_ids": [f"{task['id']}-c{i:04d}" for i in range(len(ranges))]}
