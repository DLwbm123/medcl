"""Administrator-only export from existing, trusted training runs; no training or scoring.

The private JSON config supplies the original runtime import paths, checkpoints,
case inputs and display templates. Run this in the training environment, not the UI.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


def segmentation(case, arrays, device):
    import h5py
    import torch
    import runner_core as runtime

    if "block_sizes" in case:
        model = runtime.ClassModel(block_sizes=case["block_sizes"])
    else:
        model = runtime._build_model(case["scenario"])
    model.activate_stage(case["stage"])
    model.load_state_dict(torch.load(case["checkpoint"], map_location="cpu", weights_only=True), strict=True)
    model.to(device).eval()
    with h5py.File(case["h5"], "r") as source:
        end = int(source["patient_info_train"][0]) + 1
        image = np.moveaxis(np.asarray(source["train_images"][:, :, :end]), -1, 0)
    if not np.isfinite(image).all():
        raise ValueError("Nonfinite source image")
    steps = np.maximum(1, np.ceil(np.array(image.shape) / 128).astype(int))
    slices = tuple(slice(None, None, int(step)) for step in steps)
    # Check the selected case/grid before reusing its recorded display geometry.
    from prepare_showcase import normalize
    if not np.array_equal(normalize(image[slices]), arrays["image"]):
        raise ValueError("Source case does not match its display template")
    predictions = []
    for start in range(0, len(image), 4):
        batch = torch.from_numpy(image[start:start + 4, None].astype(np.float32)).to(device)
        probabilities = runtime.zs_forward(model, batch, None)["pred_masks"]
        if not torch.isfinite(probabilities).all():
            raise ValueError("Nonfinite model output")
        prediction = probabilities.argmax(1).cpu().numpy()
        if "label_ids" in case:
            prediction = np.asarray(case["label_ids"])[prediction]
        predictions.append(prediction)
    prediction = np.concatenate(predictions)[slices]
    if prediction.shape != arrays["image"].shape or not np.isin(prediction, range(8)).all():
        raise ValueError("Invalid segmentation output")
    return {**arrays, "labels": prediction.astype(np.uint8)}


def classification(case, arrays, device):
    import torch
    from torchvision import transforms
    from PIL import Image
    from backbone.ResNet18 import resnet18

    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        payload = torch.load(case["checkpoint"], map_location="cpu", weights_only=True)
    model = resnet18(nclasses=case["classes"], in_ch=3)
    model.load_state_dict(payload.get("model_state_dict", payload.get("model")), strict=True)
    model.to(device).eval()
    operations = [transforms.Resize((case["input_size"], case["input_size"]))] if case["input_size"] != arrays["image"].shape[0] else []
    image = transforms.Compose(operations + [transforms.ToTensor()])(Image.fromarray(arrays["image"]))
    logits = model(image[None].to(device))
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite model output")
    labels = case["label_ids"]
    prediction = labels[int(logits[:, labels].argmax(1))]
    return {**arrays, "class_id": np.asarray(prediction, dtype=np.int64)}


def registration(case, arrays, device):
    import torch
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    from core.datasets.continual3d import Continual3D
    from core.methods.sgd import Sgd

    # These administrator-selected checkpoints also contain trusted RNG metadata.
    payload = torch.load(str(case["checkpoint"]), map_location="cpu", mmap=True, weights_only=False)
    if case["task"] == "oasis":
        if payload.get("next_task") != 1 or payload.get("next_step") != 0:
            raise ValueError("Expected the first-task boundary checkpoint")
    elif (payload["protocol"].get("task") != case["task"]
          or payload["protocol"].get("initialization") != "fresh_seed_42"
          or payload["protocol"].get("history_access") is not False):
        raise ValueError("Expected a single-task reference checkpoint")
    cfg = OmegaConf.load(case["config"])["source"]
    cfg.var = OmegaConf.create({}, flags={"allow_objects": True})
    cfg.var.obj_operator = SimpleNamespace(device=device, path_exp=str(Path(case["template"]).parent), task_idx=0)
    if cfg.dataset.normalization != "min-max":
        raise ValueError("Expected min-max image normalization")
    model = Sgd(cfg).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    model.before_epoch("val")
    dataset = Continual3D(cfg, task=case["task"], mode="test")
    index = 0  # Fixed source ordering, without scoring or selecting predictions.
    pair = dataset.dataset.get_image_name(index)
    if pair[0] == pair[1]:
        raise ValueError("Expected two different source images")
    sample = dataset[index]
    zooms = np.asarray(sample["headers"][0].get_zooms()[:3], dtype=float)
    if not np.allclose(zooms, sample["headers"][1].get_zooms()[:3]):
        raise ValueError("Image pair has incompatible voxel spacing")
    batch = dataset.to_device(dataset.get_batch([sample]), device)
    model({key: batch[key] for key in ("imgs", "masks", "names")})
    # OASIS's provider swaps Y/Z; the other providers retain native XYZ axes.
    axes = (1, 2, 0) if case["task"] == "oasis" else (2, 1, 0)
    result = {}
    for key, tensor in (("fixed", model.fixed_img), ("moving", model.moving_img),
                        ("registered", model.warped_moving_img)):
        volume = tensor[0, 0].cpu().numpy().transpose(axes)
        if not np.isfinite(volume).all():
            raise ValueError("Nonfinite registration output")
        result[key] = (volume.clip(0, 1) * 255).astype(np.uint8)
    case.update(pair=list(pair), pair_index=index, display_axes=list(axes),
                checkpoint_protocol=payload["protocol"])
    return {**result, "spacing": zooms[::-1], "spacing_source": np.asarray("protocol")}


def main(config):
    import torch

    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    sys.path[:0] = config["import_paths"]
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    records = []
    with torch.inference_mode():
        for case in config["cases"]:
            if Path(case["file"]).name != case["file"]:
                raise ValueError("Invalid output filename")
            with np.load(case["template"], allow_pickle=False) as template:
                arrays = {key: template[key] for key in template.files}
            result = {"segmentation": segmentation, "classification": classification,
                      "registration": registration}[config["kind"]](
                case, arrays, torch.device(config.get("device", "cpu")))
            path = output / (case["file"] + "-independent.npz")
            with path.open("xb") as stream:
                np.savez_compressed(stream, **result)
            path.chmod(0o600)
            records.append({**case, "output": path.name, "bytes": path.stat().st_size})
            print(json.dumps({"exported": path.name, "bytes": path.stat().st_size}), flush=True)
    provenance = output / (config["name"] + "-provenance.private.json")
    with provenance.open("x") as stream:
        json.dump({"source": "existing_independent_models", "scored": False,
                   "import_paths": config["import_paths"], "records": records}, stream, indent=2)
    provenance.chmod(0o600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    main(json.loads(parser.parse_args().config.read_text()))
