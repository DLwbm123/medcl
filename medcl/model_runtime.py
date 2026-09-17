"""Registered PyTorch structures. This file is also the downloadable model template.

Save build_model(architecture, classes).state_dict() with torch.save, or export
contiguous tensors with safetensors.torch.save_file. Upload weights, never this file.
"""
import re
from collections.abc import Mapping

NEURAL_ARCHITECTURES = ("resnet18-v1", "unet2d-v1", "pathmnist-resnet18-v1", "zs-domain-unet-v1")


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
        if {"conv1.weight", "linear.weight", "linear.bias"} <= keys:
            return "pathmnist-resnet18-v1"
    elif selector == "auto-segmentation-v1":
        if keys == {"weight", "bias"}:
            return "pixel-linear-v1"
        if {"down.0.0.weight", "head.weight", "head.bias"} <= keys:
            return "unet2d-v1"
        if {"backbone.Conv1.conv.0.weight", "head.conv.weight", "head.norm.running_mean"} <= keys:
            return "zs-domain-unet-v1"
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
    if architecture in ("resnet18-v1", "pathmnist-resnet18-v1"):
        from torchvision.models import resnet18
        model = resnet18(weights=None, num_classes=classes)
        if architecture == "pathmnist-resnet18-v1":
            model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        return model
    if architecture == "zs-domain-unet-v1":
        return build_domain_unet(classes)
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


def build_domain_unet(classes):
    """Inference graph matching ZScribbleSeg (Shangqi Gao, Fudan) DomainModel weights."""
    import torch
    from torch import nn
    if classes != 2:
        raise ValueError("ZS domain U-Net requires a shared binary head")

    def conv(cin, cout):
        layer = nn.Module()
        layer.conv = nn.Sequential(nn.Conv2d(cin, cout, 3), nn.ReLU(inplace=True), nn.BatchNorm2d(cout),
                                   nn.Conv2d(cout, cout, 3), nn.ReLU(inplace=True),
                                   nn.BatchNorm2d(cout, eps=1e-3, momentum=.01))
        return layer

    class DomainUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Module()
            for index, (cin, cout) in enumerate(zip((1, 64, 128, 256, 512), (64, 128, 256, 512, 1024)), 1):
                setattr(self.backbone, f"Conv{index}", conv(cin, cout))
            for index, cin, projected, skip, cout in ((4, 1024, 4, 512, 512), (3, 512, 4, 256, 256),
                                                    (2, 256, 128, 128, 128), (1, 128, 64, 64, 64)):
                layer = nn.Module()
                layer.up = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(cin, projected, 3, padding=1),
                                         nn.ReLU(inplace=True), nn.BatchNorm2d(projected, eps=1e-3, momentum=.01))
                setattr(self.backbone, f"Up{index}", layer)
                setattr(self.backbone, f"Up_conv{index}", conv(projected + skip, cout))
            self.head = nn.Module()
            self.head.conv = nn.Conv2d(64, 2, 1)
            self.head.norm = nn.BatchNorm2d(2, eps=1e-3, momentum=.01)

        def forward(self, image):
            size = image.shape[-2:]
            x = nn.functional.pad(image, (92, 92, 92, 92))
            skips = []
            for index in range(1, 6):
                if index > 1:
                    x = nn.functional.max_pool2d(x, 2)
                x = getattr(self.backbone, f"Conv{index}").conv(x)
                skips.append(x)
            for index in range(4, 0, -1):
                x = getattr(self.backbone, f"Up{index}").up(x)
                skip = skips[index - 1]
                top, left = (skip.shape[-2] - x.shape[-2]) // 2, (skip.shape[-1] - x.shape[-1]) // 2
                skip = skip[:, :, top:top + x.shape[-2], left:left + x.shape[-1]]
                x = getattr(self.backbone, f"Up_conv{index}").conv(torch.cat((x, skip), dim=1))
            probabilities = self.head.norm(self.head.conv(x)).softmax(1)
            if probabilities.shape[-2:] != size:
                probabilities = nn.functional.interpolate(probabilities, size=size, mode="bilinear", align_corners=False)
                probabilities = probabilities / probabilities.sum(1, keepdim=True).clamp_min(1e-12)
            return probabilities

    return DomainUNet()


def pathmnist_weights(state):
    """Map the native CIFAR-style ResNet names to the identical torchvision graph."""
    import torch
    aliases = {"0": "conv1", "1": "bn1", "3": "layer1", "4": "layer2", "5": "layer3", "6": "layer4"}
    result = {}
    for key, value in state.items():
        canonical = key
        if key.startswith("_features."):
            _, index, suffix = key.split(".", 2)
            canonical = aliases.get(index, "invalid") + "." + suffix
        elif key.startswith("classifier."):
            canonical = "linear." + key.split(".", 1)[1]
        if canonical != key:
            if canonical not in state or not torch.equal(value, state[canonical]):
                raise ValueError("模型重复参数不一致")
            continue
        if key.startswith("linear."):
            canonical = "fc." + key.split(".", 1)[1]
        canonical = canonical.replace(".shortcut.", ".downsample.")
        result[canonical] = value
    return result


def predict_neural(architecture, weights, images, active_classes, all_classes, options):
    import numpy as np
    import torch
    import torch.nn.functional as F
    native_pathmnist = architecture == "pathmnist-resnet18-v1"
    native_domain = architecture == "zs-domain-unet-v1"
    torch.set_num_threads(8 if native_pathmnist or native_domain else 1)
    if native_domain and (images.ndim != 3 or images.shape[1:] != (256, 256) or all_classes != [0, 1]):
        raise ValueError("ZS domain U-Net expects 256×256 single-channel binary segmentation")
    if native_pathmnist:
        if images.dtype != np.uint8 or images.shape[1:] != (28, 28, 3) or all_classes != list(range(9)):
            raise ValueError("该 ResNet 权重适配仅支持 PathMNIST 28×28 RGB、9 类协议")
        if options not in ({}, {"input_size": 0, "normalization": "unit"}):
            raise ValueError("PathMNIST ResNet 使用固定 PIL 128×128 与 ToTensor 预处理")
        weights = pathmnist_weights(weights)
        from PIL import Image
        from torchvision import transforms
        transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    model = build_model(architecture, max(all_classes) + 1)
    model.load_state_dict(weights, strict=True)
    model.eval()
    predictions = []
    allowed = torch.tensor(active_classes, dtype=torch.int64)
    with torch.inference_mode():
        # Bounded batches keep original image geometry and avoid all-test-set logits.
        batch_size = 1 if native_domain else 4
        for start in range(0, len(images), batch_size):
            raw = images[start:start + batch_size]
            x = torch.from_numpy(raw.astype(np.float32))
            if native_pathmnist:
                x = torch.stack([transform(Image.fromarray(item)) for item in raw])
            elif architecture == "resnet18-v1":
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
