"""Operator-only full-test export from trusted, existing segmentation checkpoints.

RUN_CONFIG points to a private JSON with runtime_root, output, benchmarks and
models. Models specify name, benchmark_id, scenario, checkpoint, stage, device.
Run in the original training environment. Never import this from the upload UI.
"""
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from medcl.benchmarks import read_task, validate_benchmark


def main():
    config = json.loads(Path(os.environ["RUN_CONFIG"]).read_text())
    sys.path.insert(0, config["runtime_root"])
    import runner_core as runtime
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    benchmarks = {b["id"]: validate_benchmark(b) for b in config["benchmarks"]}
    torch.set_num_threads(4)
    for item in config["models"]:
        destination = output / (item["name"] + ".npz")
        if destination.exists():
            raise FileExistsError(destination)
        benchmark = benchmarks[item["benchmark_id"]]
        model = runtime._build_model(item["scenario"])
        # Task heads are created incrementally; materialize every trained head before strict loading.
        for stage in range(item["stage"] + 1):
            model.activate_stage(stage)
        model.load_state_dict(torch.load(item["checkpoint"], map_location="cpu", weights_only=True), strict=True)
        model.eval().to(item["device"])
        arrays, counts = {}, {}
        with torch.inference_mode():
            for index, task in enumerate(benchmark["tasks"]):
                data = read_task(benchmark, task, include_targets=False)
                chunks = []
                for start in range(0, len(data["images"]), 4):
                    images = torch.from_numpy(data["images"][start:start + 4, None].astype(np.float32)).to(item["device"])
                    task_id = index if item["scenario"] == "organ" else None
                    probabilities = runtime.zs_forward(model, images, task_id)["pred_masks"]
                    if not torch.isfinite(probabilities).all():
                        raise ValueError("nonfinite reference predictions")
                    chunks.append(probabilities.argmax(1).cpu().numpy().astype(np.uint8))
                prediction = np.concatenate(chunks)
                if prediction.shape != data["images"].shape or not np.isin(prediction, task["all_classes"]).all():
                    raise ValueError("reference output does not match the fixed protocol")
                arrays[task["id"] + "__ids"] = data["sample_ids"]
                arrays[task["id"] + "__pred"] = prediction
                counts[task["id"]] = {"slices": len(prediction), "cases": len(data["ranges"])}
                print(item["name"], task["id"], counts[task["id"]], flush=True)
        with destination.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        (output / (item["name"] + ".private.json")).write_text(json.dumps({"model": item, "counts": counts}, indent=2))
        del model
        torch.cuda.empty_cache()
        print("COMPLETE", item["name"], destination.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
