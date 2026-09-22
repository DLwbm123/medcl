"""Gallery navigation and reference rendering must never create scored jobs."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

from medcl import showcase
from medcl.storage import list_jobs
from medcl_cornerstone import unpack_envelope


class ShowcaseTest(unittest.TestCase):
    def test_static_slices_respect_each_plane_spacing_without_editing_pixels(self):
        volume = np.arange(4 * 6 * 8, dtype=np.uint8).reshape(4, 6, 8)
        before = volume.copy()
        # Physical lengths are Z=12, Y=12, X=8, so orthogonal planes differ.
        for axis, shape in enumerate(((8, 5), (8, 5), (6, 6))):
            pixels = np.take(volume, 1, axis=axis)
            self.assertEqual(showcase.physical_slice(pixels, [3, 2, 1], axis).shape, shape)
            rgb = np.repeat(pixels[..., None], 3, axis=-1)
            self.assertEqual(showcase.physical_slice(rgb, [3, 2, 1], axis).shape, (*shape, 3))
        np.testing.assert_array_equal(volume, before)
        pixels = volume[0]
        self.assertIs(showcase.physical_slice(pixels, [3, 1, 1], 0), pixels)
        with self.assertRaises(ValueError): showcase.physical_slice(pixels, [0, 1, 1], 0)

    def test_hdf5_spacing_survives_sampling_and_envelope(self):
        import h5py
        from scripts.prepare_showcase import segmentation, display_spacing
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.h5"
            image = np.arange(258 * 130 * 4, dtype=np.float32).reshape(258, 130, 4)
            with h5py.File(path, "w") as f:
                f["train_images"], f["train_labels"] = image, np.ones_like(image)
                f["patient_info_train"] = [3]
            arrays = segmentation(path, weak=False, voxel_spacing=[3.6, .5, .8])
            # HWN -> ZYX, then strides [1,3,2], not isotropic voxel indices.
            np.testing.assert_allclose(arrays["spacing"], [3.6, 1.5, 1.6])
            header, _ = unpack_envelope(showcase.volume_envelope("segmentation-full", arrays))
            np.testing.assert_allclose(header["spacing_zyx"], [3.6, 1.5, 1.6])
            self.assertEqual(header["spacing_source"], "protocol")
            self.assertEqual(str(segmentation(path, weak=False)["spacing_source"]), "index-space-default")
            for invalid in ([0, 1, 1], [-1, 1, 1], [float("nan"), 1, 1], [float("inf"), 1, 1], [1, 2], [True] * 3):
                with self.assertRaises(ValueError): display_spacing(np.ones(3), invalid)

    def test_gallery_uses_asset_spacing_without_calibration_controls(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"MEDCL_STATE_DIR": directory}), \
                patch.object(showcase, "render_volume", return_value={}) as render:
            folder = Path(directory) / "showcase"
            folder.mkdir()
            arrays = {"image": np.zeros((6, 8, 10), dtype=np.uint8),
                      "labels": np.ones((6, 8, 10), dtype=np.uint8), "spacing": np.array([1., 2., 2.])}
            np.savez_compressed(folder / "segmentation-domain-T1.npz", **arrays)
            app = AppTest.from_string("import streamlit as st\nfrom medcl.showcase import render\nrender(st, 'segmentation-full')").run()
            next(c for c in app.radio if c.label == "展示视图").set_value("三维浏览").run()
            self.assertFalse(app.exception)
            self.assertFalse(app.expander)
            self.assertFalse(app.checkbox)
            self.assertFalse(app.number_input)
            self.assertFalse(app.warning)
            header, volumes = unpack_envelope(render.call_args.args[0])
            self.assertEqual(header["spacing_zyx"], [1., 2., 2.])
            self.assertEqual(header["spacing_source"], "index-space-default")
            np.testing.assert_array_equal(volumes["image"], arrays["image"])
            np.testing.assert_array_equal(volumes["prediction"], arrays["labels"])
            np.testing.assert_array_equal(showcase.load_example("segmentation-full", task_id="T1")["spacing"], [1, 2, 2])
            next(c for c in app.radio if c.label == "展示视图").set_value("切片对比").run()
            self.assertFalse(app.warning)
            self.assertFalse(app.expander)
            self.assertFalse(app.number_input)

    def test_classification_preparation_requires_native_resolution(self):
        from scripts.prepare_showcase import classification
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            image = np.arange(128 * 128 * 3, dtype=np.uint8).reshape(128, 128, 3)
            np.save(folder / "train_images.npy", image[None])
            np.save(folder / "train_labels.npy", np.array([[2]], dtype=np.uint8))
            result = classification(folder)
            np.testing.assert_array_equal(result["image"], image)
            self.assertEqual(result["class_id"], 2)
            np.save(folder / "train_images.npy", np.zeros((1, 28, 28, 3), dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "resolution"):
                classification(folder)

    def test_prepare_cardiac_keeps_all_labels_and_first_case(self):
        import h5py
        from scripts.prepare_showcase import segmentation
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cardiac.h5"
            image = np.arange(10 * 12 * 16, dtype=np.float32).reshape(10, 12, 16)
            labels = np.zeros(image.shape, dtype=np.int64)
            labels[2:8, 3:10, :8] = np.arange(8)
            labels[:, :, 8:] = 99  # A later case must not be read.
            with h5py.File(path, "w") as f:
                f["test_images"], f["test_labels"] = image, labels
                f["patient_info_test"] = [7, 15]
            result = segmentation(path, weak=False, cardiac=True)
            np.testing.assert_array_equal(result["labels"], np.moveaxis(labels[:, :, :8], -1, 0))
            np.testing.assert_array_equal(result["spacing"], [1, 1, 1])
            calibrated = segmentation(path, weak=False, cardiac=True, voxel_spacing=[4, .7, .8])
            np.testing.assert_allclose(calibrated["spacing"], [4, .7, .8])
            self.assertEqual(str(calibrated["spacing_source"]), "protocol")
            with h5py.File(path, "r+") as f:
                f["test_labels"][:, :, 7] = 0
            with self.assertRaisesRegex(ValueError, "all seven"):
                segmentation(path, weak=False, cardiac=True)

    def test_reference_gallery_navigation_and_pixels(self):
        from medcl.classification_showcase import DATASETS, card_html
        from medcl.showcase_tasks import SCENARIOS, task_specs
        import base64, io
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                "MEDCL_STATE_DIR": directory, "MEDCL_CONFIG": str(Path(directory) / "absent.json"),
                "MEDCL_SHOW_DEMOS": "0"}):
            folder = Path(directory) / "showcase"
            folder.mkdir()
            cases = {}
            for dataset, info in DATASETS.items():
                specs = task_specs("classification", dataset=dataset)
                self.assertEqual(len(specs), {"pathmnist": 4, "skin": 3, "hyperkvasir": 10}[dataset])
                for index, spec in enumerate(specs):
                    cases[spec["file"]] = {"image": np.full((info["size"], info["size"], 3), 50 + index * 10, dtype=np.uint8),
                                           "class_id": np.asarray(spec["classes"][0])}
                cases[info["file"]] = cases[specs[0]["file"]]
            image = np.full((8, 16, 16), 80, dtype=np.uint8)
            labels = np.zeros_like(image)
            labels[2:6, 4:12, 4:12] = 1
            for scenario in SCENARIOS:
                specs = task_specs("segmentation", scenario=scenario)
                self.assertEqual(len(specs), {"domain": 6, "class": 3, "task": 4}[scenario])
                for index, spec in enumerate(specs):
                    dense = labels * (1 + spec["shift"])
                    arrays = {"image": image + index, "labels": dense, "spacing": np.ones(3)}
                    cases[spec["file"]] = arrays
                    sparse = np.full_like(image, 255)
                    sparse[3, 6, 6:9] = 1 + spec["shift"]
                    sparse[3, 2, 2:5] = 0
                    cases[spec["file"].replace("segmentation-", "segmentation-weak-", 1)] = {
                        **arrays, "scribble": sparse, "scribble_ignore": np.asarray(255, dtype=np.uint8)}
            for index, spec in enumerate(task_specs("registration")):
                cases[spec["file"]] = {"fixed": image + index, "moving": image // 2 + index,
                                       "registered": image + index, "spacing": np.ones(3)}
            self.assertEqual(len(cases), 50)  # 47 task entries and three legacy classification cards.
            # Deliberately different test outputs make accidental reference reuse visible.
            for name, arrays in list(cases.items()):
                if "-T" not in name:
                    continue
                predicted = {key: value.copy() for key, value in arrays.items()}
                field = "class_id" if "class_id" in arrays else "labels" if "labels" in arrays else "registered"
                predicted[field] = arrays[field] + 1 if field == "class_id" else np.zeros_like(arrays[field])
                cases[f"{name}-independent"] = predicted
            cardiac = np.zeros_like(image)
            for label in range(1, 8): cardiac[:, label * 2:label * 2 + 2, 3:12] = label
            cases["segmentation-cardiac"] = {"image": image, "labels": cardiac, "spacing": np.ones(3)}
            for name, arrays in cases.items(): np.savez_compressed(folder / f"{name}.npz", **arrays)
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
            visits = 0
            for name in (key for key in showcase.EXAMPLES if key != "segmentation-cardiac"):
                next(b for b in app.button if b.key == f"home-showcase-{name}").click().run()
                kind = showcase.EXAMPLES[name]["kind"]
                groups = list(DATASETS) if kind == "classification" else list(SCENARIOS) if kind == "segmentation" else ["task"]
                for group in groups:
                    if kind == "classification":
                        next(c for c in app.segmented_control if c.label == "影像类型").set_value(group).run()
                    elif kind == "segmentation":
                        next(c for c in app.selectbox if c.label == "持续学习场景").set_value(group).run()
                    dataset = group if kind == "classification" else "pathmnist"
                    scenario = group if kind == "segmentation" else "domain"
                    specs = task_specs(kind, scenario=scenario, dataset=dataset)
                    task_control = next(c for c in app.radio if c.label == "持续学习任务")
                    self.assertEqual(len(task_control.options), len(specs))
                    for spec in specs:
                        next(c for c in app.radio if c.label == "持续学习任务").set_value(spec["id"]).run()
                        for sample in (1, 2):
                            next(c for c in app.radio if c.label == "示例").set_value(sample).run()
                            self.assertFalse(app.exception)
                            self.assertFalse(app.info)
                            self.assertEqual(list_jobs(), [])
                            self.assertTrue(any(f"当前 {spec['id']}：{spec['name']}" in c.value for c in app.caption))
                            source = showcase.load_example(name, dataset=dataset, scenario=scenario, task_id=spec["id"], sample=sample)
                            filename = spec["file"].replace("segmentation-", "segmentation-weak-", 1) if name == "segmentation-weak" else spec["file"]
                            if sample == 2:
                                filename += "-independent"
                            self.assertEqual(next(c for c in app.radio if c.label == "示例").options, ["示例 1", "示例 2"])
                            for field, expected in cases[filename].items(): np.testing.assert_array_equal(source[field], expected)
                            if kind == "classification":
                                html = card_html(dataset, source)
                                self.assertIn(html, [item.proto.body for item in app.get("html")])
                                pixels = np.asarray(Image.open(io.BytesIO(base64.b64decode(html.split("data:image/png;base64,")[1].split('"')[0]))))
                                np.testing.assert_array_equal(pixels, source["image"])
                                self.assertIn(DATASETS[dataset]["labels"][int(source["class_id"])], html)
                            else:
                                with patch.object(showcase, "render_volume", return_value={}) as render:
                                    next(c for c in app.radio if c.label == "展示视图").set_value("三维浏览").run()
                                    header, packed = unpack_envelope(render.call_args.args[0])
                                    self.assertEqual("-sample-2" in render.call_args.kwargs["key"], sample == 2)
                                next(c for c in app.radio if c.label == "展示视图").set_value("切片对比").run()
                                self.assertEqual(header["context"]["task_id"], spec["id"])
                                self.assertNotIn("score", header["context"])
                                if kind == "segmentation": np.testing.assert_array_equal(packed["prediction"], source["labels"])
                                else: np.testing.assert_array_equal(packed["registered"], source["registered"])
                            visits += 1
                if name == "segmentation-full":
                    next(c for c in app.selectbox if c.label == "持续学习场景").set_value("class").run()
                    next(c for c in app.radio if c.label == "持续学习任务").set_value("T3").run()
                    next(c for c in app.radio if c.label == "示例").set_value(1).run()
                    next(c for c in app.checkbox if c.label == "查看最终七类完整心脏").check().run()
                    self.assertFalse(app.exception)
                    self.assertTrue(any("最终七类" in c.value for c in app.caption))
                next(c for c in app.segmented_control if c.label == "主导航").set_value("首页").run()
            self.assertEqual(visits, 94)
            next(b for b in app.button if b.label == "进入分割任务").click().run()
            next(c for c in app.segmented_control if c.label == "增量场景").set_value("类别增量").run()
            next(b for b in app.button if b.label == "打开可视化").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(next(c for c in app.selectbox if c.label == "持续学习场景").value, "class")
            # Ignore 255 remains unchanged even when label 4 is a real foreground.
            sparse = cases["segmentation-weak-class-T2"]["scribble"][3]
            pixels = showcase.overlay(image[3], sparse, 1, scribble=True, ignore=255)
            np.testing.assert_array_equal(pixels[0, 0], [80, 80, 80])
            np.testing.assert_array_equal(pixels[6, 6], showcase.COLORS[3])
            np.testing.assert_array_equal(pixels[2, 2], [220, 226, 235])
            for kwargs in ({"task_id": "../private"}, {"scenario": "unknown", "task_id": "T1"}, {"dataset": "../private"},
                           {"task_id": "T1", "sample": "../private"}, {"task_id": "T1", "sample": True}, {"sample": 2}):
                with self.assertRaises(ValueError): showcase.load_example("classification", **kwargs)
            np.savez_compressed(folder / "classification-pathmnist-T1.npz", image=np.zeros((128,128,3), dtype=np.uint8), class_id=8)
            with self.assertRaisesRegex(ValueError, "task mismatch"):
                showcase.load_example("classification", task_id="T1")
            (folder / "segmentation-class-T1-independent.npz").unlink()
            with self.assertRaises(FileNotFoundError):
                showcase.load_example("segmentation-full", scenario="class", task_id="T1", sample=2)
            next(c for c in app.radio if c.label == "示例").set_value(2).run()
            self.assertFalse(app.exception)
            self.assertTrue(any("此素材暂不可用" in c.value for c in app.info))
            next(c for c in app.radio if c.label == "示例").set_value(1).run()
            self.assertFalse(app.info)
            (folder / "segmentation-domain-T6.npz").unlink()
            with self.assertRaises(FileNotFoundError): showcase.load_example("segmentation-full", task_id="T6")

    def test_sparse_prefix_checks_geometry_and_preserves_global_ids(self):
        from scripts.prepare_showcase import sparse_prefix
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sparse.npz"
            array = np.full((4, 5, 6), -100, dtype=np.int16)
            array[1, 2, 3] = 7
            np.savez_compressed(path, annotations=array)
            np.testing.assert_array_equal(sparse_prefix(path, 2, array.shape), array[:2])
            with self.assertRaisesRegex(ValueError, "geometry"):
                sparse_prefix(path, 2, (4, 6, 5))


if __name__ == "__main__":
    unittest.main()
