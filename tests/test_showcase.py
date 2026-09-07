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
            with h5py.File(path, "r+") as f:
                f["test_labels"][:, :, 7] = 0
            with self.assertRaisesRegex(ValueError, "all seven"):
                segmentation(path, weak=False, cardiac=True)

    def test_reference_gallery_navigation_and_pixels(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                "MEDCL_STATE_DIR": directory, "MEDCL_CONFIG": str(Path(directory) / "absent.json"),
                "MEDCL_SHOW_DEMOS": "0"}):
            folder = Path(directory) / "showcase"
            folder.mkdir()
            image = np.full((8, 16, 16), 80, dtype=np.uint8)
            labels = np.zeros_like(image)
            labels[2:6, 4:12, 4:12] = 1
            scribble = np.full_like(image, 4)
            scribble[3, 6, 6:9] = 1
            scribble[3, 2, 2:5] = 0
            volumes = {"image": image, "labels": labels, "spacing": np.ones(3)}
            cases = {"classification": {"image": np.full((128, 128, 3), 100, dtype=np.uint8), "class_id": np.asarray(8)},
                     "segmentation-full": volumes,
                     "segmentation-weak": {**volumes, "scribble": scribble},
                     "registration": {"fixed": image, "moving": image // 2, "registered": image, "spacing": np.ones(3)}}
            cardiac = np.zeros_like(image)
            for label in range(1, 8):
                cardiac[:, label * 2:label * 2 + 2, 3:12] = label
            cases["segmentation-cardiac"] = {**volumes, "labels": cardiac}
            cases["classification-skin"] = {"image": np.full((128, 128, 3), 130, dtype=np.uint8), "class_id": np.asarray(1)}
            cases["classification-hyperkvasir"] = {"image": np.full((224, 224, 3), 170, dtype=np.uint8), "class_id": np.asarray(7)}
            for name, arrays in cases.items():
                np.savez_compressed(folder / f"{name}.npz", **arrays)
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
            for name in (key for key in showcase.EXAMPLES if key != "segmentation-cardiac"):
                next(b for b in app.button if b.key == f"home-showcase-{name}").click().run()
                self.assertFalse(app.exception)
                self.assertIn("示例展示", app.title[0].value)
                self.assertEqual(len(app.metric), 0)
                self.assertEqual(list_jobs(), [])
                text = " ".join(str(e.value) for group in (app.caption, app.title, app.markdown) for e in group)
                self.assertNotIn("GT", text)
                self.assertNotIn("ground truth", text.lower())
                self.assertNotIn("预测", text)
                if name == "classification":
                    from medcl.classification_showcase import DATASETS, card_html
                    for dataset, info in DATASETS.items():
                        next(c for c in app.segmented_control if c.label == "影像类型").set_value(dataset).run()
                        self.assertFalse(app.exception)
                        source = showcase.load_example(name, dataset=dataset)
                        html = card_html(dataset, source)
                        self.assertIn(info["name"], html)
                        self.assertIn(info["labels"][int(source["class_id"])], html)
                        import base64, io
                        from PIL import Image
                        pixels = np.asarray(Image.open(io.BytesIO(base64.b64decode(html.split("data:image/png;base64,")[1].split('"')[0]))))
                        np.testing.assert_array_equal(pixels, source["image"])
                        self.assertEqual(list_jobs(), [])
                    next(c for c in app.segmented_control if c.label == "影像类型").set_value(None).run()
                    self.assertFalse(app.exception)
                    self.assertIn(next(c for c in app.segmented_control if c.label == "影像类型").value, DATASETS)
                if name != "classification":
                    header, arrays = unpack_envelope(showcase.volume_envelope(name, showcase.load_example(name)))
                    self.assertNotIn("score", header["context"])
                    self.assertNotIn("stage", header["context"])
                    next(s for s in app.selectbox if s.label == "切面").set_value(1).run()
                    self.assertFalse(app.exception)
                if name == "segmentation-full":
                    next(s for s in app.selectbox if s.label == "示例病例").set_value("segmentation-cardiac").run()
                    self.assertFalse(app.exception)
                    self.assertIn("心脏七结构", app.title[0].value)
                    header, packed = unpack_envelope(showcase.volume_envelope("segmentation-cardiac", showcase.load_example("segmentation-cardiac")))
                    np.testing.assert_array_equal(packed["prediction"], cardiac)
                    self.assertEqual(set(np.unique(packed["prediction"])), set(range(8)))
                    pixels = showcase.overlay(image[0], cardiac[0], 1)
                    for label in range(1, 8):
                        np.testing.assert_array_equal(pixels[label * 2, 4], showcase.COLORS[label - 1])
                    self.assertEqual(len(set(map(tuple, showcase.COLORS))), 7)
                next(c for c in app.segmented_control if c.label == "主导航").set_value("首页").run()
            # Sparse unlabelled pixels remain intact; background scribbles are visible.
            pixels = showcase.overlay(image[3], scribble[3], 1, scribble=True)
            np.testing.assert_array_equal(pixels[0, 0], [80, 80, 80])
            np.testing.assert_array_equal(pixels[6, 6], showcase.COLORS[0])
            np.testing.assert_array_equal(pixels[2, 2], [220, 226, 235])
            np.testing.assert_array_equal(showcase.checkerboard(image[0], image[0] // 2)[0, [0, 12]], [80, 40])
            with self.assertRaises(ValueError):
                showcase.load_example("../private")
            for filename, shape, label in [("classification", (28, 28, 3), 0), ("classification-skin", (128, 128, 3), 6),
                                            ("classification-hyperkvasir", (128, 128, 3), 7)]:
                np.savez_compressed(folder / f"{filename}.npz", image=np.zeros(shape, dtype=np.uint8), class_id=label)
                dataset = {"classification": "pathmnist", "classification-skin": "skin", "classification-hyperkvasir": "hyperkvasir"}[filename]
                with self.assertRaises(ValueError):
                    showcase.load_example("classification", dataset=dataset)
            np.savez_compressed(folder / "classification.npz", image=np.zeros((128, 128, 3)), class_id=100)
            with self.assertRaises(ValueError):
                showcase.load_example("classification")


if __name__ == "__main__":
    unittest.main()
