"""Label-blind examples, explicitly marked as engineering baselines."""

import io
import json

import h5py
import numpy as np
from safetensors.numpy import save

from medcl.benchmarks import read_task
from medcl.submissions import SCHEMA


def sample_manifest(benchmark: dict) -> bytes:
    tasks = {}
    for task in benchmark["tasks"]:
        if task["format"] == "h5":
            with h5py.File(task["path"], "r") as data:
                n = data["test_images"].shape[-1]
                shape = [n, *data["test_images"].shape[:2]]
            ids = [f"{task['id']}-s{i:06d}" for i in range(n)]
        elif task["format"] == "npy":
            labels = np.load(task["labels_path"], mmap_mode="r", allow_pickle=False).reshape(-1)
            indices = np.flatnonzero(np.isin(labels, task["classes"])) if task.get("filter_classes", True) else np.arange(len(labels))
            ids = [f"{task['id']}-s{i:06d}" for i in indices]
            shape = [len(ids)]
        else:
            data = read_task(benchmark, task, include_targets=False)
            ids = data["sample_ids"].tolist()
            shape = list(data["images"].shape) if benchmark["kind"] != "classification" else [len(ids)]
        tasks[task["id"]] = {"sample_ids": ids, "prediction_shape": shape, "classes": task.get("all_classes", []),
                             "spacing": task.get("spacing"), "coordinate_system": task.get("coordinate_system")}
    return json.dumps({"benchmark": benchmark["id"], "version": benchmark["version"],
                       "note": "样本索引，不含隐藏标签。任务顺序和阶段在界面选择，无需手写配置清单。", "tasks": tasks},
                      ensure_ascii=False, indent=2).encode("utf-8")


def baseline_predictions(benchmark: dict, task_ids: list[str], active_classes: list[int] | None = None) -> dict:
    tasks = {}
    for task in benchmark["tasks"]:
        if task["id"] not in task_ids:
            continue
        data = read_task(benchmark, task, include_targets=False)
        images = data["images"]
        if benchmark["kind"] == "segmentation":
            prediction = (images > 0.5).astype(np.uint8)
        elif benchmark["kind"] == "classification":
            # No training, label inspection or test-label-based parameter selection.
            if benchmark.get("synthetic"):
                logits = np.column_stack([images[:, 0], images[:, 1], np.zeros(len(images))])
                allowed = np.asarray(active_classes or [0, 1, 2])
                prediction = allowed[logits[:, allowed].argmax(1)]
            else:
                prediction = np.full(len(images), (active_classes or [0])[0], dtype=np.int64)
        else:
            prediction = images.astype(np.float64)  # Identity: submitted moving points unchanged.
        tasks[task["id"]] = {"sample_ids": data["sample_ids"], "predictions": prediction}
    return {"schema": SCHEMA, "tasks": tasks}


def pack_predictions(document: dict, as_json: bool = False) -> bytes:
    if as_json:
        return json.dumps(document, ensure_ascii=False, allow_nan=False, default=lambda x: x.tolist()).encode("utf-8")
    arrays = {}
    for key, task in document["tasks"].items():
        arrays[f"{key}__ids"] = np.asarray(task["sample_ids"])
        pred = np.asarray(task["predictions"])
        arrays[f"{key}__pred"] = pred.astype(np.uint8) if pred.dtype.kind in "iu" and pred.max(initial=0) < 256 and pred.min(initial=0) >= 0 else pred
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def example_weights(kind: str, features: int = 2, classes: int = 3) -> bytes:
    if kind == "segmentation":
        weights = {"weight": np.array([[0], [1]], dtype=np.float32), "bias": np.array([0, -0.5], dtype=np.float32)}
    elif kind == "registration":
        weights = {"offset": np.zeros(2, dtype=np.float32)}
    else:
        w = np.zeros((classes, features), dtype=np.float32)
        if classes == 3 and features == 2:
            w[0, 0], w[1, 1] = 1, 1
        weights = {"weight": w, "bias": np.zeros(classes, dtype=np.float32)}
    return save(weights, metadata={"purpose": "engineering-check-only-not-trained"})


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from medcl.benchmarks import catalog, readiness, state_path
    parser = argparse.ArgumentParser(description="Label-blind engineering examples; NOT trained-model results")
    parser.add_argument("--benchmark", default="demo-classification")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--all-stages", action="store_true")
    args = parser.parse_args()
    b = next((b for b in catalog() if b["id"] == args.benchmark), None)
    if b is None or not readiness(b)[0]:
        raise SystemExit("Benchmark unavailable")
    output = args.output or state_path() / "examples" / b["id"]
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    order = [t["id"] for t in b["tasks"]]
    for stage in (range(1, len(order) + 1) if args.all_stages else [len(order)]):
        active = sorted({c for t in b["tasks"][:stage] for c in t.get("classes", [])})
        doc = baseline_predictions(b, order[:stage], active)
        suffix = "npz" if b["kind"] == "segmentation" else "json"
        path = output / f"stage-{stage:02d}.{suffix}"
        with path.open("xb") as f:
            f.write(pack_predictions(doc, as_json=suffix == "json"))
        print(f"Engineering baseline (NOT trained): {path}")
    if b.get("synthetic"):
        path = output / "untrained.safetensors"
        with path.open("xb") as f:
            f.write(example_weights(b["kind"]))
        print(f"Untrained architecture fixture: {path}")
