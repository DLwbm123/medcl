"""One evaluation process. All scores come from submitted predictions or vetted inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import traceback

import numpy as np

from medcl.benchmarks import check_assets, read_task
from medcl.metrics import continual_summary, federated_summary, score_cases
from medcl.sandbox import model_predictions
from medcl.storage import encode, get_job, job_dir, update_job
from medcl.submissions import align_predictions, load_predictions, required_tasks


def aggregate(cases: list[dict], kind: str) -> float | None:
    usable = [c for c in cases if c["score"] is not None and c["n_samples"] > 0]
    if not usable:
        return None
    return float(np.average([c["score"] for c in usable],
                            weights=[c["n_samples"] for c in usable] if kind == "classification" else None))


def evaluate(job_id: str, root: Path | None = None) -> dict:
    job = get_job(job_id, root)
    if job is None:
        raise ValueError("评测不存在")
    config, folder = job["config"], job_dir(job_id, root)
    private = json.loads((folder / "private.json").read_text(encoding="utf-8"))
    check_assets(private["assets"])
    benchmark = private["benchmark"]
    kind, order = benchmark["kind"], config["order"]
    tasks = {t["id"]: t for t in benchmark["tasks"]}
    cells, cases, federated, distributions = [], [], [], []
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
                case.update(stage=stage, task_id=task_id, case_id=data["case_ids"][index],
                            client_id=f"C{index % config['clients'] + 1:02d}")
            cases.extend(scored)
            client_scores, counts = [], []
            for client_id in matrices:
                selected = scored if client_id == "global" else [c for c in scored if c["client_id"] == client_id]
                value = aggregate(selected, kind)
                n_samples = sum(c["n_samples"] for c in selected)
                cell = {"stage": stage, "task_id": task_id, "client_id": client_id, "score": value,
                        "n_samples": n_samples, "n_cases": sum(c["n_samples"] > 0 for c in selected),
                        "metric": benchmark["metric"], "direction": benchmark["direction"], "unit": benchmark["unit"],
                        "reason": "无测试样本，不可计算" if value is None else "实测"}
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
                                    "n_samples": n_samples, "n_cases": cell["n_cases"]}
                    if kind == "classification":
                        labels = [data["target"][a:b] for c in selected for a, b in [data["ranges"][c["case_index"]]]]
                        flat = np.concatenate(labels) if labels else np.asarray([], dtype=int)
                        distribution["class_counts"] = {str(k): int((flat == k).sum()) for k in task["all_classes"]}
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
    result = {"job_id": job_id, "config": config, "cells": cells, "cases": cases,
              "matrices": matrices, "continual": summaries, "federated": federated,
              "distributions": distributions, "warnings": warnings}
    temporary = folder / "result.pending.json"
    temporary.write_text(encode(result), encoding="utf-8")
    temporary.replace(folder / "result.json")
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
