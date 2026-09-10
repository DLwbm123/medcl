import io
import json
import unittest
import xml.etree.ElementTree as ET
from PIL import Image, ImageStat

from medcl import benchmark_charts as c
from medcl.reference_results import *
from medcl.reference_exports import csv_bytes, export_view, json_bytes, render_chart


def values(node):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "values" and isinstance(v, list) and v and isinstance(v[0], dict):
                yield from v
            else: yield from values(v)
    elif isinstance(node, list):
        for v in node: yield from values(v)


class BenchmarkCharts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load_reference()
        cls.rows = select_rows(cls.registry["main_results"], "Domain-CL")

    def test_specs_encode_reported_values_domains_and_missing(self):
        rows = c.metric_rows(self.rows, "A-Dice")
        self.assertEqual(rows[0]["method_id"], "ParIso-PNN")
        self.assertEqual((rows[0]["low"], rows[0]["high"]), (.787, .791))
        self.assertTrue(c.ranking(self.rows).to_dict()["layer"][0]["encoding"]["x"]["scale"]["zero"])
        self.assertIn({"baseline": 0}, list(values(c.ranking(self.rows, "BWTR").to_dict())))
        self.assertIn({"baseline": 1}, list(values(c.ranking(self.rows, "RMA").to_dict())))
        self.assertLess(len(c.ranking(self.rows, sd=False).to_dict()["layer"]), len(c.ranking(self.rows).to_dict()["layer"]))
        ring = list(values(c.composition(self.rows).to_dict()))
        self.assertEqual(sum(r["count"] for r in ring if "count" in r), 13)
        heat = list(values(c.cross_scene(self.registry["main_results"], ["Repl-DER++"], "A-Dice").to_dict()))
        missing = next(v for v in heat if v.get("scenario") == "Organ-CL")
        self.assertIsNone(missing["mean"])
        self.assertEqual(missing["text"], "未报告")
        for mode in ("SAM", "SAM-LoRA"):
            self.assertEqual(c.sam_heatmap(self.registry, mode).to_dict()["layer"][0]["encoding"]["color"]["scale"]["domain"], [0, 1])
        delta = c.sam_heatmap(self.registry, "LoRA−SAM").to_dict()
        lo, hi = delta["layer"][0]["encoding"]["color"]["scale"]["domain"]
        self.assertEqual(lo, -hi)
        self.assertEqual(sum(v.get("mean", 0) < 0 for v in values(delta)), 4)
        curve = list(values(c.sam_curve(self.registry, "T4").to_dict()))
        self.assertEqual([r["mean"] for r in curve if r.get("method_id") == "SAM"], [.728,.663,.887,.875,.826,.632])

    def test_all_chart_forms_render_and_offline_exports_are_real(self):
        class_rows = select_rows(self.registry["main_results"], "Class-CL")
        chart_set = {"排名": c.ranking(self.rows), "构成": c.composition(self.rows),
            "BWTR": c.ranking(self.rows, "BWTR"), "RMA": c.ranking(self.rows, "RMA"),
            "E-FWT": c.ranking(self.rows, "E-FWT"), "稳定性": c.tradeoff(self.rows),
            "MPE": c.tradeoff(self.rows, "MPE"), "DRR": c.tradeoff(self.rows, "DRR"),
            "WCD": c.dumbbell(class_rows), "跨场景": c.cross_scene(self.registry["main_results"], [r["method_id"] for r in self.rows], "RMA"),
            "顺序": c.order_sd(self.registry["order_robustness"], "A-Dice"),
            "SAM": c.sam_heatmap(self.registry, "SAM"), "LoRA": c.sam_heatmap(self.registry, "SAM-LoRA"),
            "差值": c.sam_heatmap(self.registry, "LoRA−SAM"), "阶段": c.sam_curve(self.registry, "T4")}
        payload = dict(scope={"scenario": "Domain-CL", "view": "验收全部图形"}, records=flatten(self.rows))
        report = export_view(payload, chart_set, [(ASSETS / "domain_memory_size_adice.png", "原图；无逐点数据")])
        for name, image in report["charts"].items():
            with self.subTest(name=name):
                svg = ET.fromstring(image["svg"])
                self.assertGreater(len(list(svg.iter())), 20)
                png = Image.open(io.BytesIO(image["png"])).convert("RGB")
                self.assertGreater(min(png.size), 300)
                self.assertGreater(max(ImageStat.Stat(png).stddev), 15)
        html = report["html"].decode()
        self.assertEqual(html.count("data:image/png;base64,"), len(chart_set) + 1)
        self.assertNotIn("<script", html)
        self.assertNotIn("src='http", html)
        self.assertIn("论文历史结果", html)
        self.assertNotIn("/Users/", html)
        self.assertEqual(json.loads(report["json"])["records"], payload["records"])
        self.assertTrue(report["csv"].startswith(b"\xef\xbb\xbf"))
        self.assertIn("-0.242", report["csv"].decode("utf-8-sig"))
        self.assertIn("'", csv_bytes([{"method": "=SUM(1)", "mean": -.2}]).decode("utf-8-sig"))
        with self.assertRaises(ValueError): json_bytes({"mean": float("nan")})
        # Single-method zero SD and a frontier of one point are also valid.
        render_chart(c.ranking([self.rows[-2]], "BWTR"))
        render_chart(c.tradeoff(self.rows[:1], "MPE"))


if __name__ == "__main__":
    unittest.main()
