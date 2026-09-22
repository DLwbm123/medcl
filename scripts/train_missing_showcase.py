"""Train the two missing independent dense examples in a trusted external runtime.

RUN_CONFIG points to a private JSON containing runtime_root, data_root, output,
task (T3 or T4), and device. Use --check before launching a fresh run.
The original HDF5 split and label conversion are preserved.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    check = parser.parse_args().check
    config = json.loads(Path(os.environ["RUN_CONFIG"]).read_text())
    if config["task"] not in ("T3", "T4"):
        raise ValueError("This launch is limited to the missing T3/T4 examples")
    sys.path.insert(0, config["runtime_root"])
    import h5py
    import numpy as np
    import torch
    import runner_core as runtime

    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    task_index = int(config["task"][1:]) - 1
    task = runtime.TASKS["organ"][task_index]
    data_root = Path(config["data_root"])
    source = data_root / task.folder / task.filename
    audit = {"task": task.code, "source": str(source.resolve()), "splits": {}}
    with h5py.File(source, "r") as handle:
        for split in ("train", "val", "test"):
            image, label = handle[f"{split}_images"], handle[f"{split}_labels"]
            ends = np.asarray(handle[f"patient_info_{split}"])
            if image.shape != label.shape or image.shape[:2] != (256, 256):
                raise ValueError(f"Invalid {split} grid")
            if (ends.ndim != 1 or not len(ends) or not np.isfinite(ends).all()
                    or not np.equal(ends, ends.astype(np.int64)).all()
                    or ends[-1] != image.shape[2] - 1):
                raise ValueError(f"Invalid {split} patient boundaries")
            lengths = np.diff(np.r_[-1, ends])
            if (lengths < 0).any() or (split != "train" and (lengths == 0).any()):
                raise ValueError(f"Invalid {split} patient intervals")
            audit["splits"][split] = {
                "slices": image.shape[2], "patient_entries": len(ends),
                "empty_entries": int((lengths == 0).sum()),
            }

    class CachedSlices(runtime.H5Slices):
        def _open(self):
            # Native HDF5 RAM cache avoids repeated strided reads over NFS.
            # Two read-only handles per run use up to twice the source file size.
            if self._handle is None:
                self._handle = h5py.File(self.path, "r", driver="core", backing_store=False)
            return self._handle

    runtime.H5Slices = CachedSlices
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device(config["device"])
    if check:
        dataset = CachedSlices(source, "train", full_supervision=True)
        batch = [dataset[i] for i in np.linspace(0, len(dataset) - 1, 4, dtype=int)]
        image = torch.stack([row[0] for row in batch]).to(device)
        label = torch.stack([row[1] for row in batch]).to(device)
        if not torch.isfinite(image).all() or not ((label == 0) | (label == 1)).all():
            raise ValueError("Invalid sampled training image or dense label")
        model = runtime._build_model("organ").to(device)
        model.activate_stage(task_index)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)
        target = runtime.native_target(label, 2)
        probabilities = runtime.zs_forward(model, image, task_index)["pred_masks"]
        loss = runtime.pce_loss(probabilities, target)
        if probabilities.shape != (4, 2, 256, 256) or not torch.isfinite(loss):
            raise ValueError("Invalid forward output or loss")
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise FloatingPointError("Nonfinite gradients")
        optimizer.step()
        # Match the exporter: activate the task head, then strictly load weights.
        restored = runtime._build_model("organ")
        restored.activate_stage(task_index)
        restored.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()}, strict=True)
        dataset.close()
        audit.update(check="passed", loss=float(loss.detach()),
                     gpu=torch.cuda.get_device_name(device),
                     peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        print(json.dumps(audit), flush=True)
        return

    args = SimpleNamespace(
        output=Path(config["output"]), data_root=data_root, sparse_root=None,
        independent_task=task_index + 1, independent_supervision="full",
        independent_skip_test=False, method="pce-sequential", seed=42,
        epochs_per_task=80, batch_size=4, workers=0, lr=0.03,
        validate_every=math.ceil(audit["splits"]["train"]["slices"] / 4),
        max_train_batches=None, pce_loss_weight=1.0, zs_global_weight=0.0,
        zs_spatial_loss_weight=0.0, zs_spatial_warmup_epochs=60, zs_gd_loss=False,
    )
    print(json.dumps({"status": "starting", "data": audit, "epochs": 80,
                      "selection": "validation foreground Dice", "test_for_selection": False}), flush=True)
    runtime._run_independent_references(args, runtime.TASKS["organ"], device, "organ")


if __name__ == "__main__":
    main()
