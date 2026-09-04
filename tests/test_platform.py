"""Runnable end-to-end and trust-boundary checks. All test data are synthetic."""

import copy
import csv
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from medcl.benchmarks import demo_protocol, read_task
from medcl.examples import baseline_predictions, example_weights, pack_predictions
from medcl.infer import predict
from medcl.reports import aggregate_report, compatibility, report_csv, report_html, report_json
from medcl.runner import evaluate
from medcl.sandbox import sandbox_available
from medcl.storage import claim_job, connection, get_job, initialize, job_dir, update_job
from medcl.submissions import align_predictions, inspect_upload, load_predictions, submit
from medcl.worker import run_worker


class PlatformChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="medcl-test-")
        self.root = initialize(Path(self.temp.name))
        self.b = demo_protocol("classification")
        self.order = ["T1", "T2", "T3"]

    def tearDown(self):
        self.temp.cleanup()

    def payload(self, stage, *, order=None, b=None, as_json=True):
        b, order = b or self.b, order or self.order
        tasks = {t["id"]: t for t in b["tasks"]}
        active = sorted({c for tid in order[:stage] for c in tasks[tid]["classes"]})
        doc = baseline_predictions(b, order[:stage], active)
        return {"stage": stage, "name": f"stage.{ 'json' if as_json else 'npz'}", "data": pack_predictions(doc, as_json)}

    def queue(self, uploads=None, **options):
        kwargs = {"method": "synthetic test", "order": self.order, "uploads": uploads or [self.payload(3)],
                  "mode": "predictions", "architecture": None, "clients": 3, "evaluate_unseen": False, "root": self.root}
        kwargs.update(options)
        return submit(self.b, **kwargs)

    def test_final_predictions_worker_reports_no_training_logs(self):
        jid = self.queue()
        self.assertEqual(get_job(jid, self.root)["status"], "queued")
        run_worker(self.root, once=True)
        job = get_job(jid, self.root)
        self.assertEqual(job["status"], "completed", job["message"])
        result = job["result"]
        self.assertEqual(result["matrices"]["global"][:2], [[None] * 3] * 2)
        self.assertIsNone(result["continual"]["global"]["BWT"]["value"])
        self.assertIsNone(result["continual"]["global"]["Forgetting"]["value"])
        self.assertEqual(sum(c["n_samples"] for c in result["cells"] if c["client_id"] == "global"), 45)
        for export in (report_csv(result), report_json(result), report_html(result)):
            self.assertTrue(export)
            self.assertNotIn(str(self.root).encode(), export)
            self.assertNotIn(b"labels_path", export)
        json.loads(report_json(result))
        public = json.loads(aggregate_report([result]))["runs"][0]
        self.assertNotIn("cases", public)
        self.assertTrue(all(c["client_id"] == "global" for c in public["global_cells"]))
        rows = list(csv.DictReader(io.StringIO(report_csv(result).decode("utf-8-sig"))))
        self.assertEqual(len(rows), 3 * 3 * 4)
        missing = [r for r in rows if r["stage"] == "1"]
        self.assertTrue(all(r["score"] == "" and r["n_samples"] == "" and r["reason"] == "未提交该阶段" for r in missing))

    def test_custom_order_and_missing_stage(self):
        order = ["T3", "T1", "T2"]
        jid = self.queue([self.payload(1, order=order), self.payload(3, order=order)], order=order)
        result = evaluate(jid, self.root)
        self.assertEqual(result["config"]["order"], order)
        matrix = result["matrices"]["global"]
        self.assertIsNotNone(matrix[0][0])
        self.assertEqual(matrix[0][1:], [None, None])
        self.assertEqual(matrix[1], [None] * 3)
        self.assertIsNone(result["continual"]["global"]["BWT"]["value"])
        self.assertIsNone(result["continual"]["global"]["Forgetting"]["value"])

    def test_segmentation_and_registration_e2e(self):
        for kind in ("segmentation", "registration"):
            with self.subTest(kind=kind):
                b = demo_protocol(kind)
                jid = submit(b, method="synthetic", order=self.order, uploads=[self.payload(3, b=b, as_json=False)],
                             mode="predictions", architecture=None, clients=4, evaluate_unseen=False,
                             training_supervision="weak" if kind == "segmentation" else None, root=self.root)
                result = evaluate(jid, self.root)
                self.assertIsNotNone(result["continual"]["global"]["Final average"]["value"])
                self.assertTrue(result["visualizations"])
                for preview in result["visualizations"]:
                    self.assertFalse(Path(preview["file"]).is_absolute())
                    with np.load(job_dir(jid, self.root) / preview["file"], allow_pickle=False) as archive:
                        self.assertNotIn("target", archive.files)
                        self.assertNotIn("fixed", archive.files)
                        expected = {"original", "overlay"} if kind == "segmentation" else {"moving", "prediction"}
                        self.assertEqual(set(archive.files), expected)
                if kind == "segmentation":
                    self.assertEqual(result["config"]["training_supervision"], "weak")
                    empty = [c for c in result["cells"] if c["client_id"] == "C04"]
                    self.assertTrue(all(c["score"] is None and c["n_samples"] == 0 for c in empty))
                    self.assertTrue(all("benchmark_mean" in c for c in result["cases"]))
                else:
                    self.assertAlmostEqual(result["matrices"]["global"][-1][0], np.sqrt(5), places=5)

    def test_immutable_config_and_comparison_gates(self):
        jid = self.queue()
        config = get_job(jid, self.root)["config"]
        with self.assertRaises(sqlite3.IntegrityError), connection(self.root) as db:
            db.execute("UPDATE jobs SET config='{}' WHERE id=?", (jid,))
        alternate = copy.deepcopy(config)
        alternate["order"] = list(reversed(config["order"]))
        self.assertNotEqual(compatibility(config), compatibility(alternate))
        alternate = copy.deepcopy(config)
        alternate["method"] = "another algorithm"
        self.assertEqual(compatibility(config), compatibility(alternate))
        alternate["training_supervision"] = "weak"
        self.assertNotEqual(compatibility(config), compatibility(alternate))

    def test_illegal_payloads_rejected_before_queue(self):
        for name, data, mode in (("x.pth", b"pickle", "model"), ("x.py", b"print(1)", "model"),
                                 ("x.json", b'{"schema":"medcl.predictions.v1","tasks":{},"x":NaN}', "predictions"),
                                 ("x.json", b'{"schema":"medcl.predictions.v1","tasks":{},"tasks":{}}', "predictions")):
            with self.subTest(name=name, data=data), self.assertRaises(ValueError):
                inspect_upload(name, data, mode, "linear-classifier-v1")
        bad = io.BytesIO()
        with zipfile.ZipFile(bad, "w") as z:
            z.writestr("../T1__pred.npy", b"bad")
            z.writestr("T1__ids.npy", b"bad")
        with self.assertRaises(ValueError):
            inspect_upload("bad.npz", bad.getvalue(), "predictions")
        obj = io.BytesIO()
        np.savez(obj, T1__ids=np.asarray(["T1-s000000"]), T1__pred=np.asarray([{"evil": True}], dtype=object))
        with self.assertRaises(ValueError):
            inspect_upload("object.npz", obj.getvalue(), "predictions")
        with self.assertRaises(ValueError):
            self.queue(order=["T1", "T1", "T3"])
        with self.assertRaises(ValueError):
            self.queue(evaluate_unseen=True)
        with self.assertRaises(ValueError):
            self.queue([self.payload(3), self.payload(3)])
        with self.assertRaises(ValueError):
            self.queue(training_supervision="weak")

    def test_mapping_checks_and_failed_job_visibility(self):
        data = read_task(self.b, self.b["tasks"][0])
        ids, target = data["sample_ids"], data["target"]
        entry = {"ids": ids[::-1], "pred": target[::-1]}
        np.testing.assert_array_equal(align_predictions(entry, ids, target.shape, [0, 1, 2], "classification"), target)
        for entry in ({"ids": np.repeat(ids[:1], len(ids)), "pred": target}, {"ids": ids, "pred": np.full_like(target, 7)}):
            with self.assertRaises(ValueError):
                align_predictions(entry, ids, target.shape, [0, 1, 2], "classification")
        wrong = self.payload(3)
        doc = json.loads(wrong["data"])
        doc["tasks"]["T1"]["sample_ids"][0] = "not-a-test-sample"
        wrong["data"] = json.dumps(doc).encode()
        jid = self.queue([wrong])
        run_worker(self.root, once=True)
        job = get_job(jid, self.root)
        self.assertEqual(job["status"], "failed")
        self.assertIn("样本 ID", job["message"])
        self.assertIsNone(job["result"])

    def test_model_gate_and_weights_format(self):
        weights = example_weights("classification")
        inspect_upload("model.safetensors", weights, "model", "linear-classifier-v1")
        with self.assertRaises(ValueError):
            inspect_upload("model.safetensors", weights[:-1], "model", "linear-classifier-v1")
        with patch("medcl.sandbox.sandbox_available", return_value=False), self.assertRaisesRegex(ValueError, "隔离"):
            self.queue([{"stage": 3, "name": "model.safetensors", "data": weights}], mode="model", architecture="linear-classifier-v1")

    def test_larger_linear_head_and_finite_outputs(self):
        weights = {"weight": np.zeros((9, 2352), dtype=np.float32), "bias": np.zeros(9, dtype=np.float32)}
        images = np.full((2400, 28, 28, 3), 127, dtype=np.uint8)
        pred = predict("linear-classifier-v1", weights, images, list(range(9)), list(range(9)))
        np.testing.assert_array_equal(pred, np.zeros(2400, dtype=np.int64))
        weights["bias"][8] = 1
        pred = predict("linear-classifier-v1", weights, images, [0, 1, 2], list(range(9)))
        self.assertTrue((pred == 0).all())  # Future-class logits cannot leak through the seen-class mask.
        huge = {"weight": np.full((2, 2), np.finfo(np.float32).max), "bias": np.zeros(2, dtype=np.float32)}
        with self.assertRaises((ValueError, FloatingPointError)):
            predict("linear-classifier-v1", huge, np.full((1, 2), 2.0, dtype=np.float32), [0, 1], [0, 1])

    def test_supported_final_model_sandbox_without_logs(self):
        if not sandbox_available():
            self.skipTest("Secure model execution unavailable: prediction-only platform on this host")
        for kind, architecture in (("classification", "linear-classifier-v1"), ("segmentation", "pixel-linear-v1"), ("registration", "point-translation-v1")):
            with self.subTest(kind=kind):
                b = demo_protocol(kind)
                jid = submit(b, method="untrained synthetic weights", order=self.order,
                             uploads=[{"stage": 3, "name": "model.safetensors", "data": example_weights(kind)}],
                             mode="model", architecture=architecture, clients=3, evaluate_unseen=False, root=self.root)
                result = evaluate(jid, self.root)
                self.assertIsNotNone(result["continual"]["global"]["Final average"]["value"])
                self.assertIsNone(result["continual"]["global"]["BWT"]["value"])

    def test_html_csv_escaping_and_no_partial_claiming(self):
        jid = self.queue(method="=1+1 <script>alert(1)</script>")
        result = evaluate(jid, self.root)
        self.assertNotIn(b"<script>alert", report_html(result))
        self.assertIn(b"&lt;script&gt;", report_html(result))
        self.assertIn(b"'=1+1", report_csv(result))
        self.assertIsNone(claim_job(self.root))


if __name__ == "__main__":
    unittest.main()
