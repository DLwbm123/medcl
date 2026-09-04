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
from medcl.metrics import continual_summary, federated_summary, score_cases
from medcl.sandbox import model_predictions
from medcl.storage import encode, get_job, job_dir, update_job
from medcl.submissions import align_predictions, load_predictions, required_tasks

MAX_PREVIEW_FILE = 16 * 1024 * 1024
MAX_PREVIEW_EXPANDED = 64 * 1024 * 1024


def aggregate(cases: list[dict], kind: str) -> float | None:
    usable = [c for c in cases if c["score"] is not None and c["n_samples"] > 0]
    if not usable:
        return None
    return float(np.average([c["score"] for c in usable],
                            weights=[c["n_samples"] for c in usable] if kind == "classification" else None))


def _plane(array: np.ndarray) -> np.ndarray:
    plane = np.asarray(array)
    while plane.ndim > 2:
        plane = plane[plane.shape[0] // 2]
    return plane


def _segmentation_image(image: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image, mask = _plane(image).astype(np.float32), _plane(prediction) != 0
    low, high = np.percentile(image, (1, 99))
    gray = np.zeros(image.shape, dtype=np.uint8) if high <= low else (np.clip((image - low) / (high - low), 0, 1) * 255).astype(np.uint8)
    original = np.repeat(gray[..., None], 3, axis=-1)
    overlay = original.copy()
    overlay[mask] = (overlay[mask] * 0.4 + np.array([25, 160, 120]) * 0.6).astype(np.uint8)
    return original, overlay


def _save_previews(folder: Path, kind: str, stage: int, task_id: str, data: dict,
                   prediction: np.ndarray, scored: list[dict]) -> list[dict]:
    if kind not in ("segmentation", "registration") or not scored:
        return []
    visual_dir = folder / "visuals"
    visual_dir.mkdir(exist_ok=True, mode=0o700)
    visual_dir.chmod(0o700)
    available = [i for i, case in enumerate(scored) if case["n_samples"] > 0]
    selected = [available[i] for i in np.linspace(0, len(available) - 1, min(6, len(available)), dtype=int)]
    previews = []
    for case_index in selected:
        start, end = data["ranges"][case_index]
        if kind == "segmentation":
            areas = np.count_nonzero(prediction[start:end], axis=tuple(range(1, prediction.ndim)))
            sample = start + int(np.argmax(areas)) if np.any(areas) else start + (end - start) // 2
            arrays = dict(zip(("original", "overlay"), _segmentation_image(data["images"][sample], prediction[sample])))
        else:
            sample = start
            arrays = {"moving": np.asarray(data["images"][sample], dtype=np.float32),
                      "prediction": np.asarray(prediction[sample], dtype=np.float32)}
        filename = f"stage-{stage:02d}-{task_id}-case-{case_index:04d}.npz"
        path = visual_dir / filename
        with path.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        path.chmod(0o600)
        previews.append({"kind": kind, "stage": stage, "task_id": task_id,
                         "case_id": data["case_ids"][case_index], "case_index": int(case_index),
                         "score": scored[case_index]["score"], "file": f"visuals/{filename}"})
    return previews


def load_preview(folder: Path, reference: dict) -> dict[str, np.ndarray]:
    """Read an optional private preview without trusting its stored reference or NPZ."""
    try:
        visual_root = (Path(folder) / "visuals").resolve()
        path = (Path(folder) / reference["file"]).resolve()
        kind = reference["kind"]
        expected = {"original", "overlay"} if kind == "segmentation" else {"moving", "prediction"} if kind == "registration" else None
        if expected is None or path.parent != visual_root or path.suffix != ".npz" or not path.is_file():
            raise ValueError("可视化引用无效")
        if not 0 < path.stat().st_size <= MAX_PREVIEW_FILE:
            raise ValueError("可视化文件大小无效")
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) != len(expected) or {entry.filename for entry in entries} != {f"{key}.npy" for key in expected}:
                raise ValueError("可视化数组键无效")
            if sum(entry.file_size for entry in entries) > MAX_PREVIEW_EXPANDED:
                raise ValueError("可视化数组超过展开上限")
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) != len(expected) or set(archive.files) != expected:
                raise ValueError("可视化数组键无效")
            arrays = {key: np.array(archive[key]) for key in expected}
        first, second = arrays.values()
        if first.shape != second.shape or any(size <= 0 for size in first.shape):
            raise ValueError("可视化数组形状无效")
        if kind == "segmentation":
            if first.dtype != np.uint8 or second.dtype != np.uint8 or first.ndim != 3 or first.shape[-1] != 3:
                raise ValueError("分割预览必须是同形 RGB uint8 图像")
        elif first.dtype.kind != "f" or second.dtype.kind != "f" or first.ndim != 2 or first.shape[1] not in (2, 3) or not all(np.isfinite(array).all() for array in arrays.values()):
            raise ValueError("配准预览必须是有限二维/三维浮点坐标")
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
            pred = pred.astype(np.float64 if kind == "registration" else np.int64, copy=False)
            classes = [0, *[c for c in task["classes"] if c != 0]] if kind == "segmentation" else task.get("classes", [])
            scored = score_cases(kind, pred, data["target"], data["ranges"], classes, task.get("spacing"))
            for case in scored:
                index = case["case_index"]
                if kind == "segmentation" and case["per_class"] is not None:
                    case["benchmark_mean"] = case["score"]
                    case["background"] = case["per_class"]["0"]
                    case["score"] = float(np.mean([case["per_class"][str(c)] for c in task["classes"]]))
                case.update(stage=stage, task_id=task_id,
                            client_id=f"C{index % config['clients'] + 1:02d}")
                if kind != "classification":
                    case["case_id"] = data["case_ids"][index]
            visualizations.extend(_save_previews(folder, kind, stage, task_id, data, pred, scored))
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
