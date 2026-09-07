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
            cases = {"classification": {"image": np.full((28, 28, 3), 100, dtype=np.uint8), "class_id": np.asarray(8)},
                     "segmentation-full": volumes,
                     "segmentation-weak": {**volumes, "scribble": scribble},
                     "registration": {"fixed": image, "moving": image // 2, "registered": image, "spacing": np.ones(3)}}
            for name, arrays in cases.items():
                np.savez_compressed(folder / f"{name}.npz", **arrays)
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
            for name in showcase.EXAMPLES:
                next(b for b in app.button if b.key == f"home-showcase-{name}").click().run()
                self.assertFalse(app.exception)
                self.assertIn("示例展示", app.title[0].value)
                self.assertEqual(len(app.metric), 0)
                self.assertEqual(list_jobs(), [])
                text = " ".join(str(e.value) for group in (app.caption, app.title, app.markdown) for e in group)
                self.assertNotIn("GT", text)
                self.assertNotIn("ground truth", text.lower())
                self.assertNotIn("预测", text)
                if name != "classification":
                    header, arrays = unpack_envelope(showcase.volume_envelope(name, showcase.load_example(name)))
                    self.assertNotIn("score", header["context"])
                    self.assertNotIn("stage", header["context"])
                    next(s for s in app.selectbox if s.label == "切面").set_value(1).run()
                    self.assertFalse(app.exception)
                next(c for c in app.segmented_control if c.label == "主导航").set_value("首页").run()
            # Sparse unlabelled pixels remain intact; background scribbles are visible.
            pixels = showcase.overlay(image[3], scribble[3], 1, scribble=True)
            np.testing.assert_array_equal(pixels[0, 0], [80, 80, 80])
            np.testing.assert_array_equal(pixels[6, 6], showcase.COLORS[0])
            np.testing.assert_array_equal(pixels[2, 2], [220, 226, 235])
            np.testing.assert_array_equal(showcase.checkerboard(image[0], image[0] // 2)[0, [0, 12]], [80, 40])
            with self.assertRaises(ValueError):
                showcase.load_example("../private")
            np.savez_compressed(folder / "classification.npz", image=np.zeros((28, 28, 3)), class_id=100)
            with self.assertRaises(ValueError):
                showcase.load_example("classification")


if __name__ == "__main__":
    unittest.main()
