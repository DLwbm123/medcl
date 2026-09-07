"""Read four administrator-selected cases and stream a private tar to stdout.

Run on the asset host using its existing h5py/nibabel environment. This does not
run models, write source assets, or modify evaluation records. Redirect stdout
to a private file; source provenance must not be published.
"""

import argparse
import io
import json
from pathlib import Path
import sys
import tarfile

import h5py
import nibabel as nib
import numpy as np


def normalize(array):
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite image")
    low, high = np.percentile(array, (1, 99))
    if high <= low:
        raise ValueError("Empty intensity range")
    return (np.clip((array.astype(np.float32) - low) / (high - low), 0, 1) * 255).astype(np.uint8)


def segmentation(path, *, weak):
    with h5py.File(path, "r") as source:
        if weak:
            image, labels, scribble = (source[k][:] for k in ("image", "label", "scribble"))
        else:
            end = int(source["patient_info_train"][0]) + 1
            image = np.moveaxis(source["train_images"][:, :, :end], -1, 0)
            labels = np.moveaxis(source["train_labels"][:, :, :end], -1, 0).astype(np.int64)
        if image.shape != labels.shape or not np.isin(labels, [0, 1, 2, 3]).all():
            raise ValueError("Invalid segmentation source")
        steps = np.maximum(1, np.ceil(np.array(image.shape) / 128).astype(int))
        slices = tuple(slice(None, None, int(step)) for step in steps)
        arrays = {"image": normalize(image[slices]), "labels": labels[slices].astype(np.uint8),
                  "spacing": steps.astype(np.float32)}
        if weak:
            if scribble.shape != labels.shape or not np.isin(scribble, [0, 1, 2, 3, 4]).all():
                raise ValueError("Invalid scribble source")
            arrays["scribble"] = scribble[slices].astype(np.uint8)
        return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prostate", type=Path, required=True)
    parser.add_argument("--weak", type=Path, required=True)
    parser.add_argument("--weak-geometry", type=Path, help="Corresponding original NIfTI for voxel spacing")
    parser.add_argument("--classification", type=Path, required=True, help="PathMNIST NPY directory")
    parser.add_argument("--pair", type=Path, required=True, help="Curated OASIS pair directory")
    args = parser.parse_args()
    # Deterministic first training image; no model confidence or performance selection.
    images = np.load(args.classification / "train_images.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(args.classification / "train_labels.npy", mmap_mode="r", allow_pickle=False)
    image, label = np.asarray(images[0]), int(np.asarray(labels[0]).item())
    if image.shape != (28, 28, 3) or image.dtype != np.uint8 or not 0 <= label < 9:
        raise ValueError("Invalid classification source")
    cases = {"classification": {"image": image, "class_id": np.asarray(label)},
             "segmentation-full": segmentation(args.prostate, weak=False),
             "segmentation-weak": segmentation(args.weak, weak=True)}
    if args.weak_geometry:
        geometry = nib.load(args.weak_geometry)
        with h5py.File(args.weak, "r") as source:
            if tuple(reversed(geometry.shape)) != source["image"].shape:
                raise ValueError("Weak source geometry shape mismatch")
        cases["segmentation-weak"]["spacing"] *= np.asarray(geometry.header.get_zooms()[::-1])
        cases["segmentation-weak"]["spacing_source"] = np.asarray("protocol")
    fixed, moving = [nib.load(args.pair / f"{name}_image.nii.gz") for name in ("fixed", "moving")]
    if fixed.shape != moving.shape or not np.allclose(fixed.affine, moving.affine):
        raise ValueError("Pair is not on a common display grid")
    if len(fixed.shape) != 3 or max(fixed.shape) > 160:
        raise ValueError("Use a prepared display-grid pair")
    fixed_array = normalize(np.asarray(fixed.dataobj).transpose(2, 1, 0))
    moving_array = normalize(np.asarray(moving.dataobj).transpose(2, 1, 0))
    # Reference replay: fixed image is the target display state, NOT a warped
    # moving image, model output, or dense correspondence ground truth.
    cases["registration"] = {"fixed": fixed_array, "moving": moving_array,
                             "registered": fixed_array.copy(), "spacing": np.asarray(fixed.header.get_zooms()[::-1], dtype=np.float32),
                             "spacing_source": np.asarray("protocol")}
    provenance = {"schema": "medcl.showcase-provenance.v1", "source_host": "administrator-selected asset host",
                  "sources": {key: str(value) for key, value in vars(args).items()},
                  "classification": "First training image with original class label",
                  "segmentation-full": "First training case with complete source labels; integer truncation",
                  "segmentation-weak": "Original complete labels and original sparse scribble; 4 is unlabelled",
                  "registration": "Real OASIS pair; registered slot is fixed-image target-state reference replay, not a computed registration",
                  "geometry": "Prostate: index-space steps. OASIS and optional original weak NIfTI: header voxel spacing. No patient orientation claim.",
                  "not_model_inference": True, "not_evaluation_results": True}
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
        for name, arrays in cases.items():
            content = io.BytesIO()
            np.savez_compressed(content, **arrays)
            payload = content.getvalue()
            member = tarfile.TarInfo(f"{name}.npz")
            member.size, member.mode = len(payload), 0o600
            archive.addfile(member, io.BytesIO(payload))
        payload = json.dumps(provenance, ensure_ascii=False, indent=2).encode()
        member = tarfile.TarInfo("provenance.private.json")
        member.size, member.mode = len(payload), 0o600
        archive.addfile(member, io.BytesIO(payload))


if __name__ == "__main__":
    main()
