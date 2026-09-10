import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from medcl.benchmarks import demo_protocol
from medcl.examples import baseline_predictions, pack_predictions
from medcl.storage import initialize, list_jobs
from medcl.submissions import submit
from medcl.worker import run_worker


def control(app, group, label):
    return next(c for c in getattr(app, group) if c.label == label)


class BenchmarkUI(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="medcl-ch03-test-")
        self.root = initialize(Path(self.temp.name))
        self.env = patch.dict(os.environ, {"MEDCL_STATE_DIR": str(self.root), "MEDCL_CONFIG": str(self.root / "no-assets.json"), "MEDCL_SHOW_DEMOS": "0"})
        self.env.start()
        self.app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
        control(self.app, "segmented_control", "主导航").set_value("方法比较").run()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def test_empty_jobs_all_views_scenarios_filters_reset_exports(self):
        app = self.app
        self.assertFalse(app.exception)
        self.assertEqual(control(app, "segmented_control", "结果来源").value, "第三章基准结果")
        self.assertEqual(len(app.get("vega_lite_chart")), 2)
        for view in ["能力与资源", "跨场景", "鲁棒性", "SAM扩展", "数据与来源"]:
            control(app, "segmented_control", "第三章视图").set_value(view).run()
            self.assertFalse(app.exception, view)
            self.assertGreater(len(app.get("vega_lite_chart")), 0, view)
        control(app, "segmented_control", "第三章视图").set_value("总览").run()
        control(app, "selectbox", "主排名指标").select("E-FWT").run()
        control(app, "segmented_control", "增量场景").set_value("Class-CL").run()
        self.assertEqual(control(app, "selectbox", "主排名指标").value, "A-Dice")
        self.assertEqual(len(control(app, "multiselect", "方法（可多选）").value), 15)
        control(app, "multiselect", "方法（可多选）").set_value(["ClassCL-MiB", "ClassCL-PLOP"]).run()
        control(app, "checkbox", "显示论文 SD 误差线").uncheck().run()
        self.assertFalse(app.exception)
        control(app, "multiselect", "方法（可多选）").set_value([]).run()
        self.assertTrue(any("没有方法" in m.value for m in app.info))
        control(app, "button", "重置筛选").click().run()
        self.assertEqual(len(control(app, "multiselect", "方法（可多选）").value), 15)
        control(app, "checkbox", "显示 Non-CL / JointTrain 参照").check().run()
        self.assertEqual(len(control(app, "multiselect", "方法（可多选）").value), 17)
        control(app, "segmented_control", "增量场景").set_value("Organ-CL").run()
        self.assertNotIn("Repl-DER++", control(app, "multiselect", "方法（可多选）").options)
        control(app, "segmented_control", "第三章视图").set_value("SAM扩展").run()
        self.assertTrue(any("仅报告 Domain-CL" in m.value for m in app.info))
        control(app, "button", "切换到 Domain-CL").click().run()
        self.assertEqual(len(app.get("vega_lite_chart")), 4)
        control(app, "selectbox", "观察测试任务").select("T6").run()
        control(app, "button", "生成当前视图导出").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertGreaterEqual(len(app.download_button), 11)
        self.assertEqual(list_jobs(self.root), [])
        control(app, "segmented_control", "结果来源").set_value("平台评测结果").run()
        self.assertTrue(any("至少需要两条" in m.value for m in app.info))

    def queue(self, method, clients=3):
        benchmark = demo_protocol("classification")
        order = ["T1", "T2", "T3"]
        jid = submit(benchmark, method=method, order=order,
            uploads=[dict(stage=3, name="final.json", data=pack_predictions(baseline_predictions(benchmark, order), True))],
            mode="predictions", architecture=None, clients=clients, evaluate_unseen=False, root=self.root)
        run_worker(self.root, once=True)
        return jid

    def test_one_and_two_real_completed_test_jobs_preserve_platform_comparison(self):
        # Synthetic engineering jobs exist only in this temporary test database.
        with patch.dict(os.environ, {"MEDCL_SHOW_DEMOS": "1"}):
            first = self.queue("first")
            self.app.run()
            self.assertFalse(self.app.exception)
            self.assertEqual(len(list_jobs(self.root)), 1)
            self.assertEqual(len(self.app.get("vega_lite_chart")), 2)
            second = self.queue("second")
            control(self.app, "segmented_control", "结果来源").set_value("平台评测结果").run()
            control(self.app, "multiselect", "选择 2–4 条已完成评测").set_value([first, second]).run()
            self.assertFalse(self.app.exception)
            self.assertEqual(len(self.app.dataframe), 1)
            self.assertEqual(len(self.app.get("vega_lite_chart")), 1)
            third = self.queue("different-condition", clients=2)
            self.app.run()
            control(self.app, "multiselect", "选择 2–4 条已完成评测").set_value([first, third]).run()
            self.assertTrue(any("评测条件不一致" in e.value for e in self.app.error))
            control(self.app, "segmented_control", "结果来源").set_value("第三章基准结果").run()
            self.assertFalse(self.app.exception)
            self.assertEqual(len(list_jobs(self.root)), 3)


if __name__ == "__main__":
    unittest.main()
