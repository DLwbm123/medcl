"""Audited, standalone inference entrypoint. The sandbox receives NO test labels."""

import json
import os
import resource
import sys

# Applied before numerical libraries load. Importing the pure predictor does not change its caller's limits.
if __name__ == "__main__":
    resource.setrlimit(resource.RLIMIT_CPU, (90, 90))
    resource.setrlimit(resource.RLIMIT_FSIZE, (512 * 1024**2, 512 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from safetensors.numpy import load_file


def predict(architecture, weights, images, active_classes, all_classes=None):
    if any(not np.isfinite(value).all() for value in weights.values()):
        raise ValueError("nonfinite weights")
    if architecture == "point-translation-v1":
        offset = weights["offset"]
        if offset.shape != (images.shape[-1],):
            raise ValueError("offset dimensions do not match coordinates")
        with np.errstate(over="raise", invalid="raise"):
            return images + offset
    w, b = weights["weight"], weights["bias"]
    if w.ndim != 2 or b.shape != (w.shape[0],) or len(active_classes) < 1 or max(active_classes) >= len(b):
        raise ValueError("invalid class head")
    if all_classes is not None and w.shape[0] != max(all_classes) + 1:
        raise ValueError("model must expose the registered global output head")
    if architecture == "linear-classifier-v1":
        x = images.reshape(len(images), -1).astype(np.float32)
        if images.dtype == np.uint8:
            x /= 255.0
        if w.shape[1] != x.shape[1]:
            raise ValueError("feature dimensions do not match model")
        with np.errstate(over="raise", invalid="raise"):
            # ponytail: direct contraction avoids host BLAS flag bugs; revisit for larger audited heads.
            logits = np.einsum("nd,cd->nc", x, w, optimize=False) + b
    elif architecture == "pixel-linear-v1":
        if w.shape[1] != 1:
            raise ValueError("pixel model expects one image channel")
        with np.errstate(over="raise", invalid="raise"):
            logits = images[..., None] * w[:, 0] + b
    else:
        raise ValueError("unsupported architecture")
    if not np.isfinite(logits).all():
        raise ValueError("nonfinite logits")
    allowed = np.asarray(active_classes, dtype=np.int64)
    return allowed[np.argmax(logits[..., allowed], axis=-1)]


if __name__ == "__main__":
    manifest_path, weight_path, input_path, output_path = sys.argv[1:]
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    weights = load_file(weight_path)
    with np.load(input_path, allow_pickle=False) as data:
        pred = predict(manifest["architecture"], weights, data["images"], manifest["active_classes"], manifest["all_classes"])
    np.save(output_path, pred, allow_pickle=False)
