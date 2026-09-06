"""One evaluation process. All scores come from submitted predictions or vetted inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import traceback
import zipfile

import numpy as np

from medcl.benchmarks import asset_paths, check_assets, read_task
from medcl import VIEWER_SCHEMA_VERSION
from medcl.metrics import continual_summary, federated_summary, score_cases
from medcl.sandbox import model_predictions
from medcl.storage import encode, get_job, job_dir, update_job
from medcl.submissions import align_predictions, align_registration_volumes, load_predictions, required_tasks

MAX_PREVIEW_FILE = 16 * 1024 * 1024
MAX_PREVIEW_EXPANDED = 64 * 1024 * 1024
MAX_PREVIEW_AXIS = 128
MAX_PREVIEW_VOXELS = 1_000_000
MAX_PREVIEW_CASES = 3
SEGMENTATION_PREVIEW_SCHEMA = "medcl.segmentation-volume-preview.v1"
REGISTRATION_PREVIEW_SCHEMA = "medcl.registration-volume-preview.v1"


def aggregate(cases: list[dict], kind: str) -> float | None:
    usable = [c for c in cases if c["score"] is not None and c["n_samples"] > 0]
    if not usable:
        return None
    return float(np.average([c["score"] for c in usable],
                            weights=[c["n_samples"] for c in usable] if kind == "classification" else None))


def _preview_steps(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(shape) != 3 or any(type(size) is not int or size <= 0 for size in shape):
        raise ValueError("三维预览 shape 无效")
    steps = [max(1, int(np.ceil(size / MAX_PREVIEW_AXIS))) for size in shape]
    while np.prod([(size + step - 1) // step for size, step in zip(shape, steps)]) > MAX_PREVIEW_VOXELS:
        sizes = [(size + step - 1) // step for size, step in zip(shape, steps)]
        steps[max(range(3), key=lambda axis: (sizes[axis], -axis))] += 1
    return tuple(steps)


def _sample_volume(array: np.ndarray, steps: tuple[int, int, int]) -> np.ndarray:
    volume = np.asarray(array)
    if volume.ndim != 3 or not np.isfinite(volume).all():
        raise ValueError("三维预览必须是有限 scalar volume")
    return volume[::steps[0], ::steps[1], ::steps[2]]


def _normalize_volume(array: np.ndarray, steps: tuple[int, int, int]) -> np.ndarray:
    sampled = _sample_volume(array, steps).astype(np.float32, copy=False)
    low, high = np.percentile(sampled, (1, 99))
    if high <= low:
        return np.zeros(sampled.shape, dtype=np.uint8)
    return (np.clip((sampled - low) / (high - low), 0, 1) * 255).astype(np.uint8)


def _label_volume(array: np.ndarray, steps: tuple[int, int, int]) -> np.ndarray:
    source = np.asarray(array)
    if source.ndim != 3 or source.dtype.kind not in "iu" or np.any(source < 0) or source.max(initial=0) > 65535:
        raise ValueError("三维预测 labelmap 必须是 uint16 范围内整数")
    sampled = source[::steps[0], ::steps[1], ::steps[2]]
    return sampled.astype(np.uint8 if sampled.max(initial=0) <= 255 else np.uint16, copy=False)


def _display_geometry(task: dict, steps: tuple[int, int, int]) -> tuple[np.ndarray, str, np.ndarray, np.ndarray]:
    source = "protocol" if "voxel_spacing_zyx" in task else "index-space-default"
    spacing = np.asarray(task.get("voxel_spacing_zyx", [1, 1, 1]), dtype=np.float32) * np.asarray(steps, dtype=np.float32)
    origin = np.asarray(task.get("origin_xyz", [0, 0, 0]), dtype=np.float32)
    direction = np.asarray(task.get("direction_xyz", [1, 0, 0, 0, 1, 0, 0, 0, 1]), dtype=np.float32)
    return spacing, source, origin, direction


def _save_previews(folder: Path, kind: str, stage: int, task: dict, data: dict,
                   prediction: np.ndarray, scored: list[dict], qualitative: dict[str, np.ndarray] | None = None) -> list[dict]:
    if kind not in ("segmentation", "registration") or not scored:
        return []
    visual_dir = folder / "visuals"
    visual_dir.mkdir(exist_ok=True, mode=0o700)
    visual_dir.chmod(0o700)
    available = [i for i, case in enumerate(scored) if case["n_samples"] > 0]
    selected = [available[i] for i in np.linspace(0, len(available) - 1, min(MAX_PREVIEW_CASES, len(available)), dtype=int)]
    previews = []
    for case_index in selected:
        start, end = data["ranges"][case_index]
        if kind == "segmentation":
            image_volume, prediction_volume = data["images"][start:end], prediction[start:end]
            steps = _preview_steps(tuple(map(int, image_volume.shape)))
            spacing, spacing_source, _, _ = _display_geometry(task, steps)
            arrays = {"image_volume": _normalize_volume(image_volume, steps),
                      "prediction_volume": _label_volume(prediction_volume, steps),
                      "spacing_zyx": spacing, "spacing_source": np.asarray(spacing_source),
                      "preview_schema_version": np.asarray(SEGMENTATION_PREVIEW_SCHEMA)}
            preview_kind = "segmentation-volume"
        else:
            sample = start
            if task["format"] == "registration-volume":
                steps = _preview_steps(tuple(map(int, data["fixed_volumes"][sample].shape)))
                spacing, spacing_source, origin, direction = _display_geometry(task, steps)
                arrays = {"fixed_volume": _normalize_volume(data["fixed_volumes"][sample], steps),
                          "moving_volume": _normalize_volume(data["moving_volumes"][sample], steps),
                          "moving_points": np.asarray(data["images"][sample], dtype=np.float32),
                          "predicted_points": np.asarray(prediction[sample], dtype=np.float32),
                          "spacing_zyx": spacing, "origin_xyz": origin, "direction_xyz": direction,
                          "spacing_source": np.asarray(spacing_source),
                          "preview_schema_version": np.asarray(REGISTRATION_PREVIEW_SCHEMA)}
                if qualitative and "registered" in qualitative:
                    arrays["registered_volume"] = _normalize_volume(qualitative["registered"][sample], steps)
                if qualitative and "warped_prediction" in qualitative:
                    arrays["warped_prediction"] = _label_volume(qualitative["warped_prediction"][sample], steps)
                preview_kind = "registration-volume"
            else:
                arrays = {"moving": np.asarray(data["images"][sample], dtype=np.float32),
                          "prediction": np.asarray(prediction[sample], dtype=np.float32)}
                preview_kind = "registration"
                steps, spacing_source = (1, 1, 1), "not-applicable"
        filename = f"stage-{stage:02d}-{task['id']}-case-{case_index:04d}.npz"
        path = visual_dir / filename
        with path.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        path.chmod(0o600)
        preview_shape = list(arrays["image_volume"].shape) if preview_kind == "segmentation-volume" else (
            list(arrays["fixed_volume"].shape) if preview_kind == "registration-volume" else None)
        coordinate_mode = ("fixed-display-grid" if preview_kind == "registration-volume"
                           and "origin_xyz" in task and "direction_xyz" in task else "index-space")
        previews.append({"kind": preview_kind, "stage": stage, "task_id": task["id"],
                         "case_id": data["case_ids"][case_index], "case_index": int(case_index),
                         "score": scored[case_index]["score"], "file": f"visuals/{filename}",
                         **({"shape_zyx": preview_shape, "downsampled": any(step > 1 for step in steps),
                             "spacing_source": spacing_source, "coordinate_mode": coordinate_mode,
                             "viewer_schema_version": VIEWER_SCHEMA_VERSION}
                            if preview_shape is not None else {})})
    return previews


def load_preview(folder: Path, reference: dict) -> dict[str, np.ndarray]:
    """Read an optional private preview without trusting its stored reference or NPZ."""
    try:
        visual_root = (Path(folder) / "visuals").resolve()
        path = (Path(folder) / reference["file"]).resolve()
        kind = reference["kind"]
        exact = {
            "segmentation": {"original", "overlay"},
            "segmentation-volume": {"image_volume", "prediction_volume", "spacing_zyx", "spacing_source", "preview_schema_version"},
            "registration": {"moving", "prediction"},
        }.get(kind)
        registration_base = {"fixed_volume", "moving_volume", "moving_points", "predicted_points", "spacing_zyx",
                             "origin_xyz", "direction_xyz", "spacing_source", "preview_schema_version"}
        if (exact is None and kind != "registration-volume") or path.parent != visual_root or path.suffix != ".npz" or not path.is_file():
            raise ValueError("可视化引用无效")
        if not 0 < path.stat().st_size <= MAX_PREVIEW_FILE:
            raise ValueError("可视化文件大小无效")
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            names = {entry.filename[:-4] for entry in entries if entry.filename.endswith(".npy")}
            allowed = registration_base | {"registered_volume", "warped_prediction"}
            valid_names = names == exact if exact is not None else registration_base <= names <= allowed
            if len(names) != len(entries) or not valid_names:
                raise ValueError("可视化数组键无效")
            if sum(entry.file_size for entry in entries) > MAX_PREVIEW_EXPANDED:
                raise ValueError("可视化数组超过展开上限")
        with np.load(path, allow_pickle=False) as archive:
            names = set(archive.files)
            valid_names = names == exact if exact is not None else registration_base <= names <= allowed
            if not valid_names:
                raise ValueError("可视化数组键无效")
            arrays = {key: np.array(archive[key]) for key in names}
        if kind == "segmentation":
            first, second = arrays.values()
            if first.shape != second.shape or any(size <= 0 for size in first.shape):
                raise ValueError("可视化数组形状无效")
            if first.dtype != np.uint8 or second.dtype != np.uint8 or first.ndim != 3 or first.shape[-1] != 3:
                raise ValueError("分割预览必须是同形 RGB uint8 图像")
        elif kind == "registration":
            first, second = arrays.values()
            if (first.shape != second.shape or any(size <= 0 for size in first.shape) or first.dtype.kind != "f"
                    or second.dtype.kind != "f" or first.ndim != 2 or first.shape[1] not in (2, 3)
                    or not all(np.isfinite(array).all() for array in arrays.values())):
                raise ValueError("配准预览必须是有限二维/三维浮点坐标")
        else:
            scalar_names = ("image_volume",) if kind == "segmentation-volume" else ("fixed_volume", "moving_volume")
            shape = arrays[scalar_names[0]].shape
            if len(shape) != 3 or any(size <= 0 or size > MAX_PREVIEW_AXIS for size in shape) or np.prod(shape) > MAX_PREVIEW_VOXELS:
                raise ValueError("三维预览形状无效")
            if any(arrays[name].dtype != np.uint8 or arrays[name].shape != shape for name in scalar_names):
                raise ValueError("scalar 预览必须是同形 uint8")
            label_names = ("prediction_volume",) if kind == "segmentation-volume" else tuple(
                name for name in ("warped_prediction",) if name in arrays)
            if any(arrays[name].dtype not in (np.uint8, np.uint16) or arrays[name].shape != shape for name in label_names):
                raise ValueError("labelmap 预览必须是同形 uint8/uint16")
            if kind == "registration-volume" and "registered_volume" in arrays and (
                    arrays["registered_volume"].dtype != np.uint8 or arrays["registered_volume"].shape != shape):
                raise ValueError("registered volume 预览必须是同形 uint8")
            if arrays["spacing_zyx"].shape != (3,) or arrays["spacing_zyx"].dtype.kind != "f" or not np.isfinite(arrays["spacing_zyx"]).all() or np.any(arrays["spacing_zyx"] <= 0):
                raise ValueError("预览 spacing 无效")
            if arrays["spacing_source"].shape != () or str(arrays["spacing_source"]) not in ("protocol", "index-space-default"):
                raise ValueError("预览 spacing 来源无效")
            schema = SEGMENTATION_PREVIEW_SCHEMA if kind == "segmentation-volume" else REGISTRATION_PREVIEW_SCHEMA
            if arrays["preview_schema_version"].shape != () or str(arrays["preview_schema_version"]) != schema:
                raise ValueError("预览 schema 无效")
            if kind == "registration-volume":
                if (arrays["origin_xyz"].shape != (3,) or arrays["direction_xyz"].shape != (9,)
                        or any(arrays[name].dtype.kind != "f" or not np.isfinite(arrays[name]).all()
                               for name in ("origin_xyz", "direction_xyz", "moving_points", "predicted_points"))
                        or arrays["moving_points"].shape != arrays["predicted_points"].shape
                        or arrays["moving_points"].ndim != 2 or arrays["moving_points"].shape[1] != 3):
                    raise ValueError("配准预览几何或辅助点无效")
        return arrays
    except (EOFError, KeyError, OSError, OverflowError, ValueError, zipfile.BadZipFile):
        raise ValueError("可视化文件损坏") from None


def evaluate(job_id: str, root: Path | None = None) -> dict:
    job = get_job(job_id, root)
    if job is None:
        raise ValueError("评测不存在")
    config, folder = job["config"], job_dir(job_id, root)
    private = json.loads((folder / "private.json").read_text(encoding="utf-8"))
    benchmark = private["benchmark"]
    kind, order = benchmark["kind"], config["order"]
    tasks = {t["id"]: t for t in benchmark["tasks"]}
    frozen_assets = {record["path"]: record for record in private["assets"]}
    cells, cases, federated, distributions, visualizations = [], [], [], [], []
    matrices = {name: [[None] * len(order) for _ in order] for name in ["global", *[f"C{i + 1:02d}" for i in range(config["clients"])]]}
    total = sum(len(required_tasks(config, item["stage"])) for item in private["uploads"])
    done = 0
    for item in sorted(private["uploads"], key=lambda x: x["stage"]):
        stage = item["stage"]
        required = required_tasks(config, stage)
        path = folder / "uploads" / item["filename"]
        predictions = load_predictions(path) if config["mode"] == "predictions" else None
        if predictions is not None and set(predictions) != set(required):
            raise ValueError(f"阶段 {stage} 的任务集合与协议不符；应包含：{'、'.join(required)}")
        for task_id in required:
            update_job(job_id, message=f"评分阶段 {stage} · 任务 {task_id}", progress=done / total, root=root)
            task = tasks[task_id]
            check_assets([frozen_assets[str(path)] for path in asset_paths(task)])
            data = read_task(benchmark, task)
            if benchmark["incremental"] == "class":
                allowed = sorted({c for tid in order[:stage] for c in tasks[tid]["classes"]} | ({0} if kind == "segmentation" else set()))
            else:
                allowed = task.get("all_classes", [])
            if config["mode"] == "model":
                pred = model_predictions(config["architecture"], path, data["images"], allowed, folder, task.get("all_classes", []))
                entry = {"ids": data["sample_ids"], "pred": pred}
            else:
                entry = predictions[task_id]
            pred = align_predictions(entry, data["sample_ids"], data["target"].shape, allowed, kind)
            qualitative = {}
            extra = set(entry) - {"ids", "pred"}
            if extra:
                if kind != "registration" or task["format"] != "registration-volume":
                    raise ValueError("当前任务不接受配准体数据数组")
                qualitative = align_registration_volumes(entry, data["sample_ids"], data["fixed_volumes"].shape)
            pred = pred.astype(np.float64 if kind == "registration" else np.int64, copy=False)
            classes = [0, *[c for c in task["classes"] if c != 0]] if kind == "segmentation" else task.get("classes", [])
            spacing = task.get("spacing", [1, 1, 1]) if kind == "registration" else task.get("spacing")
            scored = score_cases(kind, pred, data["target"], data["ranges"], classes, spacing)
            for case in scored:
                index = case["case_index"]
                if kind == "segmentation" and case["per_class"] is not None:
                    case["benchmark_mean"] = case["score"]
                    case["background"] = case["per_class"]["0"]
                case.update(stage=stage, task_id=task_id,
                            client_id=f"C{index % config['clients'] + 1:02d}")
                if kind != "classification":
                    case["case_id"] = data["case_ids"][index]
            visualizations.extend(_save_previews(folder, kind, stage, task, data, pred, scored, qualitative))
            if kind != "classification":
                cases.extend(scored)
            client_scores, counts = [], []
            for client_id in matrices:
                selected = scored if client_id == "global" else [c for c in scored if c["client_id"] == client_id]
                value = aggregate(selected, kind)
                n_samples = sum(c["n_samples"] for c in selected)
                cell = {"stage": stage, "task_id": task_id, "client_id": client_id, "score": value,
                        "n_samples": n_samples,
                        "metric": benchmark["metric"], "direction": benchmark["direction"], "unit": benchmark["unit"],
                        "reason": "无测试样本，不可计算" if value is None else "实测"}
                if kind != "classification":
                    cell["n_cases"] = sum(c["n_samples"] > 0 for c in selected)
                if kind == "segmentation":
                    valid = [c for c in selected if c["score"] is not None]
                    cell["benchmark_mean"] = float(np.mean([c["benchmark_mean"] for c in valid])) if valid else None
                    cell["background"] = float(np.mean([c["background"] for c in valid])) if valid else None
                cells.append(cell)
                matrices[client_id][stage - 1][order.index(task_id)] = value
                if client_id != "global":
                    client_scores.append(value)
                    counts.append(n_samples)
                    distribution = {"stage": stage, "task_id": task_id, "client_id": client_id,
                                    "n_samples": n_samples}
                    if kind != "classification":
                        distribution["n_cases"] = cell["n_cases"]
                    distributions.append(distribution)
            federated.append({"stage": stage, "task_id": task_id,
                              **federated_summary(client_scores, counts, kind, benchmark["direction"])})
            done += 1
    summaries = {client: continual_summary(matrix, benchmark["direction"], allow_unseen=config["evaluate_unseen"])
                 for client, matrix in matrices.items()}
    warnings = ["所有缺少阶段、参照或测试样本的指标保持不可计算；不插值、不补零。",
                "没有随机初始化 / 独立训练参照，FWT / RMA 不可计算。"]
    if config["clients"] > 1:
        warnings.append("单机固定逻辑客户端评测模拟；仅聚合评分，没有联邦训练、权重聚合或通信成本。")
    if benchmark.get("synthetic"):
        warnings.insert(0, "合成工程验收样例，禁止作为科研结果引用。")
    provenance = config.get("provenance", {}).get("category", "external_predictions_unknown")
    provenance_warnings = {
        "untrained_baseline": "未训练工程基线；不得作为方法性能或科研结论。",
        "trained_model_declared": "已训练模型来源为提交者声明；平台未验证训练过程。",
        "external_predictions_unknown": "外部预测来源未知；平台只验证测试评分，未验证训练过程。",
    }
    if provenance in provenance_warnings:
        warnings.insert(0, provenance_warnings[provenance])
    result = {"result_schema_version": 2, "job_id": job_id, "config": config, "cells": cells,
              "matrices": matrices, "continual": summaries, "federated": federated,
              "distributions": distributions, "visualizations": visualizations, "warnings": warnings}
    if kind != "classification":
        result["cases"] = cases
    temporary = folder / "result.pending.json"
    temporary.write_text(encode(result), encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(folder / "result.json")
    (folder / "result.json").chmod(0o600)
    update_job(job_id, status="completed", message="评分完成，报告可下载", progress=1, result=result, root=root)
    return result


def safe_error(error: Exception) -> str:
    message = str(error)
    if isinstance(error, ValueError) and any("\u4e00" <= c <= "\u9fff" for c in message) and not any(p in message for p in ("/Users/", "/Volumes/", "/tmp/", "/home/", "\\")):
        return message[:240]
    return "文件内容、维度或评分执行未通过检查；请核对提交格式，详细原因保留在私有 worker 日志"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("job_id")
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU, (540, 540))
    resource.setrlimit(resource.RLIMIT_FSIZE, (768 * 1024**2, 768 * 1024**2))
    try:
        evaluate(args.job_id, args.state)
    except Exception as exc:
        traceback.print_exc()
        update_job(args.job_id, status="failed", message=safe_error(exc), root=args.state)
        raise SystemExit(1)
