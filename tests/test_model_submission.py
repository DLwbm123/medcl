"""Model uploads must execute the registered network, preserve class masks and stay confined."""
import io
import os
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import torch
from safetensors.torch import save

from medcl.model_runtime import build_model, pathmnist_weights
from medcl.sandbox import command_for, clean_env, model_predictions, sandbox_available
from medcl.submissions import inspect_upload, validated_model_options


class ModelSubmissionChecks(unittest.TestCase):
    def test_native_domain_weights_and_geometry_in_isolation(self):
        model = build_model("zs-domain-unet-v1", 2)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.head.norm.bias[1] = 1
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            weights = root / "final.pt"
            torch.save(model.state_dict(), weights)
            inspect_upload(weights.name, weights.read_bytes(), "model", "auto-segmentation-v1")
            pred = model_predictions("auto-segmentation-v1", weights, np.zeros((1, 256, 256), np.float32),
                                     [0, 1], root, [0, 1])
            self.assertEqual(pred.shape, (1, 256, 256))
            self.assertTrue(np.all(pred == 1))

    def test_native_pathmnist_upload_and_conflicting_alias(self):
        model = build_model("pathmnist-resnet18-v1", 9)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.fc.bias.copy_(torch.arange(9, dtype=torch.float32))
        state = {key.replace("fc.", "linear.").replace(".downsample.", ".shortcut."): value
                 for key, value in model.state_dict().items()}
        state["classifier.weight"] = state["linear.weight"].clone()
        state["classifier.bias"] = state["linear.bias"].clone()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            weights = root / "final.pt"
            torch.save({"model": state, "stage": {"task": 4}}, weights)
            inspect_upload(weights.name, weights.read_bytes(), "model", "auto-classification-v1")
            pred = model_predictions("auto-classification-v1", weights, np.zeros((1, 28, 28, 3), np.uint8),
                                     list(range(6)), root, list(range(9)), validated_model_options("auto-classification-v1"))
            self.assertEqual(pred.tolist(), [5])
        state["classifier.bias"][0] = -1
        with self.assertRaisesRegex(ValueError, "重复参数不一致"):
            pathmnist_weights(state)

    def test_neural_weights_and_class_mask_in_isolation(self):
        self.assertTrue(sandbox_available(), "model isolation must pass on the release host")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            for architecture in ("resnet18-v1", "unet2d-v1"):
                model = build_model(architecture, 3)
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.zero_()
                    (model.fc if architecture == "resnet18-v1" else model.head).bias.copy_(torch.tensor([0., 1., 2.]))
                state = model.state_dict()
                buffer = io.BytesIO()
                torch.save({"model_state_dict": state}, buffer)
                for suffix, data in (("pth", buffer.getvalue()), ("safetensors", save(state))):
                    weights = root / ("weights." + suffix)
                    weights.write_bytes(data)
                    selector = "auto-classification-v1" if architecture == "resnet18-v1" else "auto-segmentation-v1"
                    inspect_upload(weights.name, data, "model", selector)
                    images = np.zeros((2, 28, 28, 3), np.uint8) if architecture == "resnet18-v1" else np.zeros((2, 17, 19), np.float32)
                    try:
                        pred = model_predictions(selector, weights, images, [0, 1], root, [0, 1, 2], validated_model_options(selector))
                    except ValueError:
                        self.fail((root / "inference-error.txt").read_text())
                    self.assertEqual(pred.shape, (2,) if architecture == "resnet18-v1" else (2, 17, 19))
                    self.assertTrue(np.all(pred == 1), "unseen class 2 must remain masked")

    def test_unsafe_weights_and_invalid_options_are_rejected(self):
        with self.assertRaises(ValueError):
            inspect_upload("old.pth", b"pickle", "model", "resnet18-v1")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name in ("archive/data.pkl", "archive/version", "archive/code/model.py"):
                archive.writestr(name, b"x")
        with self.assertRaises(ValueError):
            inspect_upload("script.pt", buffer.getvalue(), "model", "resnet18-v1")
        with self.assertRaises(ValueError):
            validated_model_options("resnet18-v1", {"input_size": 100000, "normalization": "unit"})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            marker = root / "must-not-exist"
            class Executable:
                def __reduce__(self):
                    return os.system, (f"touch {marker}",)
            buffer = io.BytesIO()
            torch.save({"weight": torch.zeros((3, 1)), "bias": torch.zeros(3), "extra": Executable()}, buffer)
            path = root / "unsafe.pth"
            path.write_bytes(buffer.getvalue())
            inspect_upload(path.name, path.read_bytes(), "model", "auto-segmentation-v1")
            with self.assertRaisesRegex(ValueError, "隔离推理失败"):
                model_predictions("auto-segmentation-v1", path, np.zeros((1, 2, 2), np.float32), [0, 1, 2], root)
            self.assertFalse(marker.exists())

    def test_no_files_network_or_child_processes_escape(self):
        import subprocess
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            secret, output = root / "private-labels", root / "output"
            secret.write_text("private")
            output.touch()
            code = f'''import os,socket,subprocess,threading
secret={str(secret)!r}
for operation in (lambda: open(secret).read(), lambda: open(secret,'w'),
                  lambda: os.truncate(secret,0), lambda: os.chmod(secret,0o777),
                  lambda: socket.socket().connect(('127.0.0.1',9)), lambda: os.fork(),
                  lambda: subprocess.run(['/usr/bin/true'])):
    try:
        operation()
    except PermissionError:
        pass
    else:
        raise AssertionError('isolation escape')
thread=threading.Thread(target=lambda: None)
thread.start();thread.join()
open({str(output)!r},'w').write('ok')
'''
            with command_for(code, [], output) as (command, script):
                result = subprocess.run(command, input=script.encode(), capture_output=True, env=clean_env(), cwd="/", timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertEqual(secret.read_text(), "private")
            self.assertEqual(output.read_text(), "ok")


if __name__ == "__main__":
    unittest.main()
