"""Runnable end-to-end and trust-boundary checks. All test data are synthetic."""

import copy
from contextlib import closing
import csv
import io
import json
import os
from pathlib import Path
import sqlite3
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from medcl.benchmarks import _ranges, demo_protocol, read_task, readiness, validate_benchmark
from medcl.examples import baseline_predictions, example_weights, pack_predictions
from medcl.infer import predict
from medcl.reports import aggregate_report, compatible_result, compatibility, report_csv, report_html, report_json
from medcl.runner import evaluate, load_preview
from medcl.sandbox import sandbox_available
from medcl.storage import claim_job, connection, encode, get_job, initialize, job_dir
from medcl.submissions import align_predictions, inspect_upload, submit
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
        active = sorted({c for tid in order[:stage] for c in tasks[tid].get("classes", [])})
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
        self.assertEqual(result["result_schema_version"], 2)
        self.assertEqual(result["config"]["provenance"]["category"], "synthetic")
        self.assertTrue(any("合成工程验收" in warning for warning in result["warnings"]))
        self.assertNotIn("cases", result)
        self.assertTrue(all("n_cases" not in cell for cell in result["cells"]))
        self.assertTrue(all("class_counts" not in row for row in result["distributions"]))
        self.assertEqual(result["matrices"]["global"][:2], [[None] * 3] * 2)
        self.assertIsNone(result["continual"]["global"]["BWT"]["value"])
        self.assertIsNone(result["continual"]["global"]["Forgetting"]["value"])
        self.assertEqual(sum(c["n_samples"] for c in result["cells"] if c["client_id"] == "global"), 45)
        for export in (report_csv(result), report_json(result), report_html(result)):
            self.assertTrue(export)
            self.assertNotIn(str(self.root).encode(), export)
            self.assertNotIn(b"labels_path", export)
            self.assertNotIn(b"T1-s000000", export)
            self.assertNotIn(b"class_counts", export)
            self.assertNotIn(b"case_id", export)
        json.loads(report_json(result))
        public = json.loads(aggregate_report([result]))["runs"][0]
        self.assertNotIn("cases", public)
        self.assertTrue(all("client_id" not in c for c in public["global_cells"]))
        final = {cell["task_id"]: cell["score"] for cell in result["cells"] if cell["stage"] == 3 and cell["client_id"] == "global"}
        self.assertEqual(final, {"T1": 1.0, "T2": 13 / 15, "T3": 1.0})
        t2_clients = next(row for row in result["federated"] if row["stage"] == 3 and row["task_id"] == "T2")
        self.assertAlmostEqual(t2_clients["sample_weighted_accuracy"], 13 / 15)
        rows = list(csv.DictReader(io.StringIO(report_csv(result).decode("utf-8-sig"))))
        self.assertTrue({"provenance_category", "training_verified", "synthetic", "evaluator_version",
                         "viewer_schema_version"} <= set(rows[0]))
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
                self.assertEqual(stat.S_IMODE((job_dir(jid, self.root) / "visuals").stat().st_mode), 0o700)
                for preview in result["visualizations"]:
                    self.assertFalse(Path(preview["file"]).is_absolute())
                    preview_path = job_dir(jid, self.root) / preview["file"]
                    self.assertEqual(stat.S_IMODE(preview_path.stat().st_mode), 0o600)
                    with np.load(preview_path, allow_pickle=False) as archive:
                        self.assertNotIn("target", archive.files)
                        self.assertNotIn("fixed", archive.files)
                        expected = ({"image_volume", "prediction_volume", "spacing_zyx", "spacing_source", "preview_schema_version"}
                                    if kind == "segmentation" else {"moving", "prediction"})
                        self.assertEqual(set(archive.files), expected)
                    self.assertEqual(set(load_preview(job_dir(jid, self.root), preview)), expected)
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
        alternate = copy.deepcopy(config)
        alternate["evaluator_version"] = "medcl-evaluator-next"
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

    def test_protocol_schema_and_strict_hdf5_ranges(self):
        for kind in ("classification", "segmentation", "registration"):
            validate_benchmark(demo_protocol(kind))
        invalid = []
        mixed = copy.deepcopy(demo_protocol("segmentation"))
        mixed["synthetic"] = False
        invalid.append(mixed)
        real_in_synthetic = copy.deepcopy(demo_protocol("segmentation"))
        real_in_synthetic["tasks"][0].update(format="h5", path="/tmp/test.h5")
        invalid.append(real_in_synthetic)
        implicit_bool = copy.deepcopy(demo_protocol("classification"))
        implicit_bool["synthetic"] = "true"
        invalid.append(implicit_bool)
        wrong_metric = copy.deepcopy(demo_protocol("classification"))
        wrong_metric["unit"] = "percent"
        invalid.append(wrong_metric)
        missing_name = copy.deepcopy(demo_protocol("classification"))
        missing_name["class_names"].pop("2")
        invalid.append(missing_name)
        overlap = copy.deepcopy(demo_protocol("classification"))
        overlap["tasks"][1]["classes"] = [0]
        invalid.append(overlap)
        no_path = copy.deepcopy(demo_protocol("segmentation"))
        no_path["synthetic"] = False
        for task in no_path["tasks"]:
            task["format"] = "h5"
        invalid.append(no_path)
        for index, benchmark in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_benchmark(benchmark)
        pending = copy.deepcopy(no_path)
        pending.update(tasks=[], pending=True)
        validate_benchmark(pending)
        self.assertFalse(readiness(pending)[0])

        self.assertEqual(_ranges([1, 4, 7], 8), [(0, 2), (2, 5), (5, 8)])
        for ends in ([-1, 7], [1, 1, 7], [3, 2, 7], [1, 4, 6]):
            with self.subTest(ends=ends), self.assertRaises(ValueError):
                _ranges(ends, 8)

    def test_json_resource_limits_and_public_allowlist(self):
        small = {"schema": "medcl.predictions.v1", "tasks": {"T1": {"sample_ids": ["T1-s000000"], "predictions": [0]}}}
        data = json.dumps(small).encode()
        with patch("medcl.submissions.MAX_JSON", len(data) - 1), self.assertRaisesRegex(ValueError, "NPZ"):
            inspect_upload("too-large.json", data, "predictions")
        with patch("medcl.submissions.MAX_JSON_PREDICTIONS", 0), self.assertRaisesRegex(ValueError, "NPZ"):
            inspect_upload("too-many.json", data, "predictions")

        result = evaluate(self.queue(method="/Users/private user@example.com 10.1.2.3"), self.root)
        result["config"]["benchmark"].update(title="/home/private", description="10.1.2.3", source="user@example.com")
        result["config"]["provenance"]["statement"] = "/Users/secret"
        public = aggregate_report([result])
        for secret in (b"/Users", b"/home", b"10.1.2.3", b"user@example.com", b'"config"', b"client_id"):
            self.assertNotIn(secret, public)
        doc = json.loads(public)
        self.assertEqual(doc["schema"], "medcl.aggregate-report.v2")
        self.assertEqual(doc["runs"][0]["run_id"], f"run-{result['job_id'][:8]}")
        self.assertEqual(doc["runs"][0]["evaluator_version"], result["config"]["evaluator_version"])

    def test_preview_corruption_and_legacy_result_degrade_safely(self):
        benchmark = demo_protocol("segmentation")
        jid = submit(benchmark, method="preview", order=self.order,
                     uploads=[self.payload(3, b=benchmark, as_json=False)], mode="predictions",
                     architecture=None, clients=2, evaluate_unseen=False, root=self.root)
        result = evaluate(jid, self.root)
        folder = job_dir(jid, self.root)
        for name, content in (("empty.npz", b""), ("truncated.npz", b"PK\x03\x04")):
            (folder / "visuals" / name).write_bytes(content)
            with self.subTest(name=name), self.assertRaises(ValueError):
                load_preview(folder, {"kind": "segmentation", "file": f"visuals/{name}"})
        wrong = io.BytesIO()
        np.savez_compressed(wrong, unexpected=np.zeros((2, 2), dtype=np.uint8))
        (folder / "visuals" / "wrong.npz").write_bytes(wrong.getvalue())
        with self.assertRaises(ValueError):
            load_preview(folder, {"kind": "segmentation", "file": "visuals/wrong.npz"})

        old = copy.deepcopy(result)
        for key in ("result_schema_version", "visualizations", "distributions", "continual"):
            old.pop(key, None)
        compatible = compatible_result(old)
        self.assertEqual(compatible["result_schema_version"], 1)
        self.assertEqual(compatible["visualizations"], [])
        self.assertIsNone(compatible["continual"]["global"]["Final average"]["value"])
        self.assertTrue(report_json(old))
        self.assertTrue(report_html(old))

    def test_owner_only_modes_and_legacy_classification_migration(self):
        jid = self.queue()
        run_worker(self.root, once=True)
        folder = job_dir(jid, self.root)
        private_files = [self.root / "medcl.sqlite3", folder / "private.json", folder / "result.json",
                         folder / "worker.private.log", *list((folder / "uploads").iterdir())]
        private_dirs = [self.root, self.root / "jobs", folder, folder / "uploads"]
        for path in private_files:
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600, path)
        for path in private_dirs:
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700, path)
        for name in ("medcl.sqlite3-wal", "medcl.sqlite3-shm"):
            path = self.root / name
            if path.exists():
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600, path)

        legacy = get_job(jid, self.root)["result"]
        legacy["cases"] = [{"case_id": "T1-c0000", "score": 1.0}]
        legacy["distributions"][0]["class_counts"] = {"0": 1}
        with connection(self.root) as db:
            db.execute("UPDATE jobs SET result=? WHERE id=?", (encode(legacy), jid))
            db.execute("PRAGMA user_version=1")
        (folder / "result.json").write_text(encode(legacy), encoding="utf-8")
        os.chmod(folder / "result.json", 0o644)
        initialize(self.root)
        with closing(sqlite3.connect(self.root / "medcl.sqlite3")) as db:
            stored = json.loads(db.execute("SELECT result FROM jobs WHERE id=?", (jid,)).fetchone()[0])
        self.assertNotIn("cases", stored)
        self.assertNotIn("class_counts", json.dumps(stored))
        migrated = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        self.assertNotIn("cases", migrated)
        self.assertEqual(stat.S_IMODE((folder / "result.json").stat().st_mode), 0o600)


if __name__ == "__main__":
    unittest.main()
