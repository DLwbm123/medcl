"""A runnable UI regression: upload, queue, result, exports and visible errors."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from medcl.benchmarks import demo_protocol
from medcl.examples import baseline_predictions, pack_predictions
from medcl.storage import heartbeat, initialize, list_jobs
from medcl.worker import run_worker


class BrowserAppCheck(unittest.TestCase):
    def test_upload_result_navigation_and_invalid_file(self):
        with tempfile.TemporaryDirectory(prefix="medcl-ui-test-") as directory:
            root = initialize(Path(directory))
            with patch.dict(os.environ, {"MEDCL_STATE_DIR": str(root), "MEDCL_CONFIG": str(root / "no-assets.json")}):
                heartbeat(root)
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
                self.assertFalse(app.exception)
                next(button for button in app.button if button.label == "进入分类任务").click().run()
                next(button for button in app.button if button.key == "configure-demo-classification").click().run()
                app.text_input[0].set_value("UI synthetic acceptance").run()
                data = pack_predictions(baseline_predictions(demo_protocol("classification"), ["T1", "T2", "T3"]), True)
                app.file_uploader[0].set_value(("final.json", data, "application/json")).run()
                heartbeat(root)
                next(button for button in app.button if button.label == "提交并开始评测").click().run()
                self.assertFalse(app.exception)
                self.assertEqual(len(list_jobs(root)), 1)
                run_worker(root, once=True)
                next(control for control in app.segmented_control if control.label == "主导航").set_value("评测记录").run()
                self.assertFalse(app.exception)
                self.assertTrue(any("合成工程验收" in warning.value for warning in app.warning))
                self.assertEqual(len(app.download_button), 3)
                self.assertIn("—", [metric.value for metric in app.metric])
                next(control for control in app.segmented_control if control.label == "主导航").set_value("方法比较").run()
                self.assertTrue(any("至少需要两条" in notice.value for notice in app.info))
                next(control for control in app.segmented_control if control.label == "主导航").set_value("任务中心").run()
                app.text_input[0].set_value("invalid upload check").run()
                app.file_uploader[0].set_value(("bad.json", b"not-json", "application/json")).run()
                heartbeat(root)
                next(button for button in app.button if button.label == "提交并开始评测").click().run()
                self.assertFalse(app.exception)
                self.assertTrue(any("JSON 文件无效" in error.value for error in app.error))
                self.assertEqual(len(list_jobs(root)), 1)

    def test_unseen_state_resets_when_output_head_changes(self):
        with tempfile.TemporaryDirectory(prefix="medcl-ui-state-") as directory:
            root = initialize(Path(directory))
            with patch.dict(os.environ, {"MEDCL_STATE_DIR": str(root), "MEDCL_CONFIG": str(root / "no-assets.json")}):
                heartbeat(root)
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
                next(button for button in app.button if button.label == "进入分割任务").click().run()
                next(button for button in app.button if button.key == "configure-demo-segmentation").click().run()
                unseen = next(box for box in app.checkbox if box.label == "同时评测未见任务")
                unseen.check().run()
                self.assertTrue(next(box for box in app.checkbox if box.label == "同时评测未见任务").value)
                head = next(box for box in app.selectbox if box.label == "输出头与任务信息条件")
                head.select("task-specific").run()
                unseen = next(box for box in app.checkbox if box.label == "同时评测未见任务")
                self.assertFalse(unseen.value)
                self.assertTrue(unseen.disabled)

                app.text_input[0].set_value("task-specific state check").run()
                benchmark = demo_protocol("segmentation")
                data = pack_predictions(baseline_predictions(benchmark, ["T1", "T2", "T3"]), False)
                app.file_uploader[0].set_value(("final.npz", data, "application/octet-stream")).run()
                heartbeat(root)
                next(button for button in app.button if button.label == "提交并开始评测").click().run()
                self.assertFalse(app.exception)
                jobs = list_jobs(root)
                self.assertEqual(len(jobs), 1)
                self.assertFalse(jobs[0]["config"]["evaluate_unseen"])


if __name__ == "__main__":
    unittest.main()
