"""Registered PyTorch structures. This file is also the downloadable model template.

Save build_model(architecture, classes).state_dict() with torch.save, or export
contiguous tensors with safetensors.torch.save_file. Upload weights, never this file.
"""
import re
from collections.abc import Mapping


def resolve_architecture(selector, keys):
    """Recognize only registered structures; strict parameter loading still follows."""
    keys = set(keys)
    if keys and all(key.startswith("module.") for key in keys):
        keys = {key[7:] for key in keys}
    if selector == "auto-classification-v1":
        if keys == {"weight", "bias"}:
            return "linear-classifier-v1"
        if {"conv1.weight", "fc.weight", "fc.bias"} <= keys:
            return "resnet18-v1"
    elif selector == "auto-segmentation-v1":
        if keys == {"weight", "bias"}:
            return "pixel-linear-v1"
        if {"down.0.0.weight", "head.weight", "head.bias"} <= keys:
            return "unet2d-v1"
    elif selector == "auto-registration-v1" and keys == {"offset"}:
        return "point-translation-v1"
    raise ValueError("无法识别该任务支持的模型权重")


def load_weights(path):
    import torch
    if tuple(int(x) for x in re.match(r"(\d+)\.(\d+)", torch.__version__).groups()) < (2, 6):
        raise ValueError("PyTorch 2.6 or newer is required")
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(str(path), device="cpu")
    else:
        # Never retry with pickle, weights_only=False or additional safe globals.
        state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in state and isinstance(state[key], Mapping):
                state = state[key]
                break
    if not isinstance(state, Mapping) or not state or any(
        not isinstance(k, str) or type(v) is not torch.Tensor or v.layout != torch.strided
        or v.dtype not in (torch.float32, torch.int64) or not torch.isfinite(v).all()
        for k, v in state.items()
    ):
        raise ValueError("expected a finite float32/int64 tensor state_dict")
    if all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    return state


def build_model(architecture, classes):
    import torch
    from torch import nn
    if architecture == "resnet18-v1":
        from torchvision.models import resnet18
        return resnet18(weights=None, num_classes=classes)
    if architecture != "unet2d-v1":
        raise ValueError("unsupported neural architecture")

    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(),
                             nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU())

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            channels = (32, 64, 128, 256)
            self.down = nn.ModuleList(block(a, b) for a, b in zip((1, *channels[:-1]), channels))
            self.bottom = block(256, 512)
            self.up = nn.ModuleList(block(a + b, b) for a, b in zip((512, 256, 128, 64), reversed(channels)))
            self.head = nn.Conv2d(32, classes, 1)

        def forward(self, x):
            skips = []
            for layer in self.down:
                x = layer(x)
                skips.append(x)
                x = nn.functional.max_pool2d(x, 2)
            x = self.bottom(x)
            for layer, skip in zip(self.up, reversed(skips)):
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                x = layer(torch.cat((skip, x), dim=1))
            return self.head(x)

    return UNet()


def predict_neural(architecture, weights, images, active_classes, all_classes, options):
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    model = build_model(architecture, max(all_classes) + 1)
    model.load_state_dict(weights, strict=True)
    model.eval()
    predictions = []
    allowed = torch.tensor(active_classes, dtype=torch.int64)
    with torch.inference_mode():
        # Bounded batches keep original image geometry and avoid all-test-set logits.
        for start in range(0, len(images), 4):
            raw = images[start:start + 4]
            x = torch.from_numpy(raw.astype(np.float32))
            if architecture == "resnet18-v1":
                if x.ndim != 4 or x.shape[-1] not in (1, 3):
                    raise ValueError("ResNet-18 expects NHWC images")
                x = x.permute(0, 3, 1, 2)
                if x.shape[1] == 1:
                    x = x.repeat(1, 3, 1, 1)
                if raw.dtype == np.uint8:
                    x = x / 255.0
                if options["input_size"]:
                    x = F.interpolate(x, size=(options["input_size"],) * 2, mode="bilinear", align_corners=False)
                if options["normalization"] == "imagenet":
                    x = (x - x.new_tensor([.485, .456, .406])[None, :, None, None]) / x.new_tensor([.229, .224, .225])[None, :, None, None]
            else:
                if x.ndim != 3 or min(x.shape[-2:]) < 16:
                    raise ValueError("U-Net expects NHW images of at least 16 x 16")
                x = x[:, None]
            logits = model(x)
            if not torch.isfinite(logits).all():
                raise ValueError("nonfinite model output")
            predictions.append(allowed[logits[:, allowed].argmax(dim=1)].numpy())
    return np.concatenate(predictions)
