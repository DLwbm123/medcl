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
    @staticmethod
    def visible_text(app):
        groups = (app.title, app.subheader, app.caption, app.info, app.warning, app.markdown)
        return "\n".join(str(element.value) for group in groups for element in group)

    def test_thesis_visual_system_is_centralized_and_wide(self):
        # MedCL homepage module/style contract v1.
        root = Path(__file__).resolve().parents[1]
        app_source = (root / "app.py").read_text()
        home_source = (root / "medcl" / "homepage_ui.py").read_text()
        assets = root / "medcl" / "ui_assets" / "homepage"
        css = (assets / "homepage.css").read_text()
        self.assertIn('NAV = ["首页", "任务中心", "评测记录", "方法比较"]', app_source)
        self.assertIn("_home_ui.render_homepage", app_source)
        self.assertIn("_home_ui.render_header", app_source)
        self.assertIn("功能示意 · 非实验结果 · 非真实病例", home_source)
        self.assertIn("st.columns(3, gap=", home_source)
        self.assertIn("max-width:1640px", css)
        self.assertIn("__MEDICAL_BACKGROUND__", css)
        self.assertNotIn("url(https://", css)
        for filename in ("medical-background.webp", "illustration-slice.webp",
                         "illustration-overlay.webp", "illustration-view.webp", "illustration-volume.webp"):
            self.assertTrue((assets / filename).is_file(), filename)

    def test_home_ctas_use_real_navigation(self):
        with tempfile.TemporaryDirectory(prefix="medcl-home-test-") as directory:
            root = initialize(Path(directory))
            with patch.dict(os.environ, {"MEDCL_STATE_DIR": str(root), "MEDCL_CONFIG": str(root / "no-assets.json"),
                                             "MEDCL_SHOW_DEMOS": "0"}):
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
                self.assertFalse(app.exception)
                self.assertEqual(next(control for control in app.segmented_control if control.label == "主导航").value, "首页")
                next(button for button in app.button if button.key == "medcl-home-start").click().run()
                self.assertEqual(next(control for control in app.segmented_control if control.label == "主导航").value, "任务中心")
                next(control for control in app.segmented_control if control.label == "主导航").set_value("首页").run()
                next(button for button in app.button if button.label == "查看评测记录").click().run()
                self.assertEqual(next(control for control in app.segmented_control if control.label == "主导航").value, "评测记录")

    def test_model_upload_queue_and_result(self):
        import io
        import torch
        from safetensors.numpy import load
        from medcl.examples import example_weights
        from medcl.storage import get_job
        from medcl.submissions import submit
        from medcl.runner import evaluate
        with tempfile.TemporaryDirectory(prefix="evaluation-ui-") as directory:
            root = initialize(Path(directory))
            with patch.dict(os.environ, {"MEDCL_STATE_DIR": str(root), "MEDCL_CONFIG": str(root / "no-assets.json"), "MEDCL_SHOW_DEMOS": "1"}):
                heartbeat(root)
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
                next(button for button in app.button if button.label == "进入分类任务").click().run()
                next(button for button in app.button if button.key == "configure-demo-classification").click().run()
                mode = next(control for control in app.radio if control.label == "评测方式")
                self.assertEqual(mode.options, ["提交预测", "提交模型"])
                mode.set_value("提交模型").run()
                self.assertFalse(app.exception)
                self.assertFalse(app.selectbox)
                self.assertFalse(app.text_input)
                self.assertFalse(app.download_button)
                buffer = io.BytesIO()
                torch.save({key: torch.from_numpy(value.copy()) for key, value in load(example_weights("classification")).items()}, buffer)
                app.file_uploader[0].set_value(("final.pth", buffer.getvalue(), "application/octet-stream")).run()
                heartbeat(root)
                next(button for button in app.button if button.label == "提交并开始评测").click().run()
                self.assertFalse(app.exception)
                self.assertEqual(len(list_jobs(root)), 1)
                run_worker(root, once=True)
                job = get_job(list_jobs(root)[0]["id"], root)
                self.assertEqual(job["status"], "completed", job["message"])
                self.assertEqual(job["config"]["mode"], "model")
                b = demo_protocol("classification")
                pred_id = submit(b, method="equivalent prediction", order=["T1", "T2", "T3"],
                                 uploads=[{"stage": 3, "name": "final.json", "data": pack_predictions(baseline_predictions(b, ["T1", "T2", "T3"]), True)}],
                                 mode="predictions", architecture=None, clients=1, evaluate_unseen=False, root=root)
                prediction_result = evaluate(pred_id, root)
                self.assertEqual(job["result"]["continual"], prediction_result["continual"])
                next(button for button in app.button if button.label == "查看此次评测").click().run()
                self.assertFalse(app.exception)
                self.assertIn("模型提交", self.visible_text(app))

    def test_upload_result_navigation_and_invalid_file(self):
        with tempfile.TemporaryDirectory(prefix="medcl-ui-test-") as directory:
            root = initialize(Path(directory))
            with patch.dict(os.environ, {"MEDCL_STATE_DIR": str(root), "MEDCL_CONFIG": str(root / "no-assets.json"),
                                             "MEDCL_SHOW_DEMOS": "1"}):
                heartbeat(root)
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
                self.assertFalse(app.exception)
                next(button for button in app.button if button.label == "进入分类任务").click().run()
                next(button for button in app.button if button.key == "configure-demo-classification").click().run()
                data = pack_predictions(baseline_predictions(demo_protocol("classification"), ["T1", "T2", "T3"]), True)
                app.file_uploader[0].set_value(("final.json", data, "application/json")).run()
                heartbeat(root)
                next(button for button in app.button if button.label == "提交并开始评测").click().run()
                self.assertFalse(app.exception)
                self.assertEqual(len(list_jobs(root)), 1)
                run_worker(root, once=True)
                job_id = list_jobs(root)[0]["id"]
                next(control for control in app.segmented_control if control.label == "主导航").set_value("首页").run()
                next(button for button in app.button if button.key == f"home-job-{job_id}").click().run()
                self.assertEqual(app.session_state["selected_job"], job_id)
                self.assertFalse(app.exception)
                self.assertTrue(any("合成工程验收" in warning.value for warning in app.warning))
                self.assertEqual(len(app.download_button), 3)
                self.assertNotIn("—", [metric.value for metric in app.metric])
                self.assertIn("仅有最终阶段", self.visible_text(app))
                next(control for control in app.segmented_control if control.label == "主导航").set_value("方法比较").run()
                next(control for control in app.segmented_control if control.label == "结果来源").set_value("平台评测结果").run()
                self.assertTrue(any("至少需要两条" in notice.value for notice in app.info))
                next(control for control in app.segmented_control if control.label == "主导航").set_value("任务中心").run()
                app.file_uploader[0].set_value(("bad.json", b"not-json", "application/json")).run()
                heartbeat(root)
                next(button for button in app.button if button.label == "提交并开始评测").click().run()
                self.assertFalse(app.exception)
                self.assertTrue(any("JSON 文件无效" in error.value for error in app.error))
                self.assertEqual(len(list_jobs(root)), 1)

                with patch.dict(os.environ, {"MEDCL_SHOW_DEMOS": "0"}):
                    normal = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
                    self.assertFalse(normal.exception)
                    self.assertFalse(any(button.key and button.key.startswith("configure-demo-") for button in normal.button))
                    next(button for button in normal.button if button.label == "进入分割任务").click().run()
                    text = self.visible_text(normal)
                    self.assertIn("当前没有可用的评测协议", text)
                    self.assertNotIn("待接入", text)
                    self.assertNotIn("合成模拟", text)
                    self.assertFalse(any(label in text for label in ("一级 /", "二级 /", "三级 /", "四级 /", "01 /", "02 /", "03 /")))
                    next(control for control in normal.segmented_control if control.label == "主导航").set_value("评测记录").run()
                    self.assertTrue(any("还没有评测记录" in notice.value for notice in normal.info))

    def test_prediction_prefix_and_final_model_cover_six_task_protocol(self):
        import copy
        from medcl.examples import example_weights
        from medcl.storage import get_job
        benchmark = demo_protocol("segmentation")
        benchmark["tasks"] = [dict(copy.deepcopy(benchmark["tasks"][0]), id=f"T{i}", name=f"任务 {i}") for i in range(1, 7)]
        order = [task["id"] for task in benchmark["tasks"]]
        with tempfile.TemporaryDirectory(prefix="evaluation-simple-") as directory:
            root = initialize(Path(directory))
            with patch.dict(os.environ, {"MEDCL_STATE_DIR": str(root), "MEDCL_SHOW_DEMOS": "1"}), patch("medcl.benchmarks.catalog", return_value=[benchmark]):
                heartbeat(root)
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
                next(button for button in app.button if button.label == "进入分割任务").click().run()
                next(button for button in app.button if button.key == "configure-demo-segmentation").click().run()
                self.assertEqual([control.label for control in app.radio], ["评测方式"])
                self.assertEqual([control.label for control in app.selectbox], ["任务"])
                self.assertFalse(app.text_input or app.checkbox or app.toggle or app.expander or app.download_button)
                visible = self.visible_text(app)
                for removed in ("评测设置", "结果来源", "输出头", "归一化", "评测协议", "训练条件"):
                    self.assertNotIn(removed, visible)
                app.selectbox[0].set_value(4).run()
                data = pack_predictions(baseline_predictions(benchmark, order[:4]), False)
                app.file_uploader[0].set_value(("T4.npz", data, "application/octet-stream")).run()
                heartbeat(root)
                next(button for button in app.button if button.label == "提交并开始评测").click().run()
                self.assertFalse(app.exception)
                run_worker(root, once=True)
                job = get_job(list_jobs(root)[0]["id"], root)
                self.assertEqual(job["status"], "completed", job["message"])
                self.assertEqual(job["config"]["stages"], [4])
                self.assertEqual({cell["task_id"] for cell in job["result"]["cells"]}, set(order[:4]))
                self.assertEqual(job["config"]["clients"], 1)
                self.assertFalse(job["config"]["evaluate_unseen"])
                app.radio[0].set_value("提交模型").run()
                self.assertFalse(app.selectbox)
                self.assertEqual(len(app.file_uploader), 1)
                self.assertEqual(app.file_uploader[0].label, "最终模型")
                app.file_uploader[0].set_value(("final.safetensors", example_weights("segmentation"), "application/octet-stream")).run()
                heartbeat(root)
                next(button for button in app.button if button.label == "提交并开始评测").click().run()
                self.assertFalse(app.exception)
                run_worker(root, once=True)
                job = get_job(list_jobs(root)[0]["id"], root)
                self.assertEqual(job["status"], "completed", job["message"])
                self.assertEqual(job["config"]["stages"], [6])
                self.assertEqual({cell["task_id"] for cell in job["result"]["cells"]}, set(order))

    def test_required_task_segments_recover_after_deselection(self):
        with tempfile.TemporaryDirectory(prefix="medcl-ui-segment-") as directory:
            root = initialize(Path(directory))
            with patch.dict(os.environ, {"MEDCL_STATE_DIR": str(root), "MEDCL_CONFIG": str(root / "no-assets.json"),
                                             "MEDCL_SHOW_DEMOS": "1"}):
                heartbeat(root)
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
                next(button for button in app.button if button.label == "进入分割任务").click().run()
                app.session_state["scenario-segmentation"] = None
                app.session_state["segmentation-supervision"] = None
                app.session_state["scope-segmentation"] = None
                app.run()
                self.assertFalse(app.exception)
                self.assertEqual(next(control for control in app.segmented_control if control.label == "增量场景").value, "域增量")
                self.assertEqual(next(control for control in app.segmented_control if control.label == "分割监督方式").value, "全监督")
                self.assertEqual(next(control for control in app.segmented_control if control.label == "数据范围").value, "全部协议")


if __name__ == "__main__":
    unittest.main()
