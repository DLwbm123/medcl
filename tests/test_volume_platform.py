"""3D preview, protocol, and registration-volume trust-boundary checks."""

import copy
import io
import json
from pathlib import Path
import stat
import struct
import tempfile
import unittest

import h5py
import numpy as np

from medcl.benchmarks import allowed_output_heads, demo_protocol, read_task, validate_benchmark
from medcl.reports import aggregate_report, report_csv, report_html, report_json
from medcl.runner import (_display_geometry, _label_volume, _normalize_volume, _preview_steps,
                          evaluate, load_preview, MAX_PREVIEW_AXIS, MAX_PREVIEW_VOXELS)
from medcl.storage import initialize, job_dir
from medcl.submissions import align_registration_volumes, inspect_upload, submit
from medcl_cornerstone import envelope_from_preview, pack_envelope, unpack_envelope


def mutate_header(data: bytes, change) -> bytes:
    length = struct.unpack("<I", data[:4])[0]
    header = json.loads(data[4:4 + length])
    change(header)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("<I", len(encoded)) + encoded + data[4 + length:]


class EnvelopeChecks(unittest.TestCase):
    def setUp(self):
        image = np.arange(2 * 3 * 5, dtype=np.uint8).reshape(2, 3, 5)
        label = (image % 3).astype(np.uint16)
        self.data = pack_envelope(viewer_mode="segmentation",
                                  volumes=[("image", "scalar", image), ("prediction", "labelmap", label)],
                                  spacing_zyx=[4, 2, 1], spacing_source="protocol", segments=[1, 2],
                                  context={"stage": 1, "task_id": "T1", "downsampled": False})

    def test_roundtrip_preserves_asymmetric_zyx_and_dtype(self):
        header, arrays = unpack_envelope(self.data)
        self.assertEqual(header["spacing_zyx"], [4.0, 2.0, 1.0])
        self.assertEqual(arrays["image"].shape, (2, 3, 5))
        self.assertEqual(arrays["prediction"].dtype, np.dtype("<u2"))
        np.testing.assert_array_equal(arrays["image"], np.arange(30, dtype=np.uint8).reshape(2, 3, 5))

    def test_truncated_header_payload_and_trailing_bytes_rejected(self):
        for broken in (self.data[:8], self.data[:-1], self.data + b"x"):
            with self.subTest(size=len(broken)), self.assertRaises(ValueError):
                unpack_envelope(broken)

    def test_wrong_dtype_shape_length_and_overlap_rejected(self):
        mutations = [
            lambda h: h["volumes"][1].update(dtype="float64"),
            lambda h: h["volumes"][1].update(shape_zyx=[2, 3, 4]),
            lambda h: h["volumes"][1].update(byte_length=2),
            lambda h: h["volumes"][1].update(offset=0),
        ]
        for change in mutations:
            with self.subTest(change=change), self.assertRaises(ValueError):
                unpack_envelope(mutate_header(self.data, change))

    def test_only_identity_direction_is_supported(self):
        close = [1 + 5e-7, 0, 0, 0, 1, 0, 0, 0, 1]
        pack_envelope(viewer_mode="segmentation",
                      volumes=[("image", "scalar", np.zeros((1, 1, 1), dtype=np.uint8)),
                               ("prediction", "labelmap", np.zeros((1, 1, 1), dtype=np.uint8))],
                      spacing_zyx=[1, 1, 1], spacing_source="index-space-default",
                      direction_xyz=close)
        with self.assertRaisesRegex(ValueError, "identity"):
            unpack_envelope(mutate_header(self.data, lambda h: h.update(
                direction_xyz=[-1, 0, 0, 0, 1, 0, 0, 0, 1])))


class VolumePlatformChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="medcl-volume-test-")
        self.root = initialize(Path(self.temp.name) / "state")

    def tearDown(self):
        self.temp.cleanup()

    def test_segmentation_downsample_is_bounded_and_nearest_neighbor(self):
        shape = (257, 129, 65)
        image = np.linspace(-2, 3, np.prod(shape), dtype=np.float32).reshape(shape)
        labels = np.zeros(shape, dtype=np.uint16)
        labels[::7, ::5, ::3] = 2
        labels[::11, ::9, ::4] = 7
        steps = _preview_steps(shape)
        scalar = _normalize_volume(image, steps)
        sampled = _label_volume(labels, steps)
        self.assertTrue(all(size <= MAX_PREVIEW_AXIS for size in scalar.shape))
        self.assertLessEqual(scalar.size, MAX_PREVIEW_VOXELS)
        self.assertEqual(scalar.shape, sampled.shape)
        self.assertEqual(scalar.dtype, np.uint8)
        self.assertIn(sampled.dtype, (np.uint8, np.uint16))
        self.assertLessEqual(set(np.unique(sampled)), set(np.unique(labels)))

    def test_spacing_validation_and_missing_default(self):
        benchmark = demo_protocol("segmentation")
        invalid = copy.deepcopy(benchmark)
        invalid["tasks"][0]["voxel_spacing_zyx"] = [1, 0, 1]
        with self.assertRaises(ValueError):
            validate_benchmark(invalid)
        spacing, source, origin, direction = _display_geometry(benchmark["tasks"][0], (2, 3, 4))
        np.testing.assert_array_equal(spacing, [2, 3, 4])
        self.assertEqual(source, "index-space-default")
        np.testing.assert_array_equal(origin, [0, 0, 0])
        np.testing.assert_array_equal(direction, [1, 0, 0, 0, 1, 0, 0, 0, 1])

    def test_label_shift_cannot_hide_a_real_foreground_class(self):
        path = Path(self.temp.name) / "shift.h5"
        with h5py.File(path, "w") as archive:
            archive["test_images"] = np.zeros((4, 5, 2), dtype=np.float32)
            labels = np.zeros((4, 5, 2), dtype=np.float32)
            labels[1:3, 1:4, :] = 1
            archive["test_labels"] = labels
            archive["patient_info_test"] = np.array([1])
        task = {"id": "T1", "name": "shift", "format": "h5", "path": str(path),
                "classes": [1], "all_classes": [0, 1, 2], "label_shift": 1}
        benchmark = {**demo_protocol("segmentation"), "id": "shift-real", "synthetic": False,
                     "version": "shift-v1", "source": "private fixture", "allow_unseen": False,
                     "allowed_output_heads": ["shared"], "tasks": [task]}
        validate_benchmark(benchmark)
        with self.assertRaisesRegex(ValueError, "类别集合"):
            read_task(benchmark, task)
        task["classes"] = [2]
        data = read_task(benchmark, task)
        self.assertEqual(set(np.unique(data["target"])), {0, 2})

    def test_domain_class_and_task_label_space_rules(self):
        domain = demo_protocol("segmentation")
        domain["tasks"][1]["classes"] = [2]
        domain["tasks"][1]["all_classes"] = [0, 1, 2]
        with self.assertRaises(ValueError):
            validate_benchmark(domain)

        class_incremental = copy.deepcopy(demo_protocol("segmentation"))
        class_incremental["incremental"] = "class"
        for index, task in enumerate(class_incremental["tasks"], 1):
            task.update(classes=[index], all_classes=[0, 1, 2, 3])
        validate_benchmark(class_incremental)
        class_incremental["tasks"][1]["classes"] = [1]
        with self.assertRaises(ValueError):
            validate_benchmark(class_incremental)

        task_incremental = copy.deepcopy(demo_protocol("segmentation"))
        task_incremental["incremental"] = "task"
        task_incremental["allow_unseen"] = False
        task_incremental["allowed_output_heads"] = ["task-specific"]
        for index, task in enumerate(task_incremental["tasks"], 1):
            task.update(classes=[index], all_classes=[0, index])
        validate_benchmark(task_incremental)
        self.assertEqual(allowed_output_heads(task_incremental), ["task-specific"])
        task_incremental["allowed_output_heads"] = ["shared"]
        with self.assertRaises(ValueError):
            validate_benchmark(task_incremental)

    def test_submit_enforces_allowed_output_heads(self):
        benchmark = demo_protocol("classification")
        with self.assertRaisesRegex(ValueError, "输出头"):
            submit(benchmark, method="head gate", order=["T1", "T2", "T3"], uploads=[],
                   mode="predictions", architecture=None, clients=1, evaluate_unseen=False,
                   output_head="task-specific", root=self.root)

    def registration_asset(self, *, mismatched=False) -> tuple[dict, np.ndarray]:
        path = Path(self.temp.name) / ("registration-bad.npz" if mismatched else "registration.npz")
        fixed = np.arange(2 * 5 * 6 * 7, dtype=np.float32).reshape(2, 5, 6, 7)
        moving = fixed[:, :, :, :-1] if mismatched else fixed + 1
        moving_points = np.array([[[1, 2, 3], [2, 3, 4]], [[2, 1, 3], [3, 4, 2]]], dtype=np.float32)
        fixed_points = moving_points + np.array([0.5, -1, 2], dtype=np.float32)
        np.savez(path, fixed_volumes=fixed, moving_volumes=moving,
                 moving_points=moving_points, fixed_points=fixed_points)
        task = {"id": "R1", "name": "volume registration", "format": "registration-volume", "path": str(path),
                "coordinate_system": "fixed-display-grid xyz, mm", "voxel_spacing_zyx": [2.5, 1.2, 1.2],
                "origin_xyz": [0, 0, 0], "direction_xyz": [1, 0, 0, 0, 1, 0, 0, 0, 1]}
        benchmark = {"id": "reg-volume", "title": "Registration volume test", "kind": "registration",
                     "incremental": "task", "version": "private-v1", "synthetic": False,
                     "description": "private test fixture", "source": "private test fixture",
                     "metric": "TRE", "direction": "lower", "unit": "mm", "allow_unseen": False,
                     "allowed_output_heads": ["shared"], "output_semantics": "fixed-display-grid-landmarks",
                     "tasks": [task]}
        return benchmark, fixed_points

    def registration_payload(self, benchmark: dict, prediction: np.ndarray, *, registered=True,
                             warped=True, registered_value=None, warped_value=None) -> bytes:
        data = read_task(benchmark, benchmark["tasks"][0])
        arrays = {"R1__ids": data["sample_ids"], "R1__pred": prediction}
        if registered:
            arrays["R1__registered"] = data["fixed_volumes"] if registered_value is None else registered_value
        if warped:
            arrays["R1__warped_prediction"] = np.zeros(data["fixed_volumes"].shape, dtype=np.uint16) if warped_value is None else warped_value
        stream = io.BytesIO()
        np.savez(stream, **arrays)
        return stream.getvalue()

    def run_registration(self, benchmark: dict, payload: bytes, method: str) -> dict:
        jid = submit(benchmark, method=method, order=["R1"],
                     uploads=[{"stage": 1, "name": "prediction.npz", "data": payload}],
                     mode="predictions", architecture=None, clients=2, evaluate_unseen=False,
                     provenance="synthetic" if benchmark["synthetic"] else "external_predictions_unknown", root=self.root)
        return evaluate(jid, self.root)

    def test_registration_volume_asset_and_optional_preview(self):
        benchmark, target = self.registration_asset()
        validate_benchmark(benchmark)
        invalid_direction = copy.deepcopy(benchmark)
        invalid_direction["tasks"][0]["direction_xyz"][0] = -1
        with self.assertRaisesRegex(ValueError, "identity"):
            validate_benchmark(invalid_direction)
        without = self.run_registration(benchmark, self.registration_payload(
            benchmark, target, registered=False, warped=False), "landmarks only")
        with_volume = self.run_registration(benchmark, self.registration_payload(
            benchmark, target), "qualitative volume")
        self.assertEqual(without["matrices"]["global"], with_volume["matrices"]["global"])
        self.assertEqual(without["matrices"]["global"][0][0], 0.0)
        first = without["visualizations"][0]
        arrays = load_preview(job_dir(without["job_id"], self.root), first)
        self.assertEqual(first["kind"], "registration-volume")
        self.assertNotIn("fixed_points", arrays)
        self.assertNotIn("registered_volume", arrays)
        self.assertTrue({"fixed_volume", "moving_volume", "moving_points", "predicted_points"} <= set(arrays))
        full_ref = with_volume["visualizations"][0]
        full = load_preview(job_dir(with_volume["job_id"], self.root), full_ref)
        self.assertIn("registered_volume", full)
        self.assertIn("warped_prediction", full)
        header, _ = unpack_envelope(envelope_from_preview(full_ref, full))
        self.assertEqual(header["viewer_mode"], "registration")
        self.assertNotIn("fixed_points", json.dumps(header))
        preview_path = job_dir(with_volume["job_id"], self.root) / full_ref["file"]
        self.assertEqual(stat.S_IMODE(preview_path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(preview_path.parent.stat().st_mode), 0o700)
        self.assertFalse(Path(full_ref["file"]).is_absolute())

        exports = [report_json(with_volume), report_csv(with_volume), report_html(with_volume), aggregate_report([with_volume])]
        for exported in exports:
            self.assertNotIn(b"fixed_volume", exported)
            self.assertNotIn(b"fixed_points", exported)
            self.assertNotIn(str(benchmark["tasks"][0]["path"]).encode(), exported)

    def test_generated_registration_volume_is_explicitly_synthetic(self):
        benchmark, target = self.registration_asset()
        benchmark["id"] = "generated-reg-volume"
        benchmark["synthetic"] = True
        benchmark["source"] = "deterministic generated fixture"
        validate_benchmark(benchmark)
        result = self.run_registration(benchmark, self.registration_payload(benchmark, target), "generated fixture")
        self.assertEqual(result["config"]["provenance"]["category"], "synthetic")
        self.assertTrue(result["config"]["benchmark"]["synthetic"])

    def test_registration_volume_shape_and_submission_arrays_are_strict(self):
        bad, _ = self.registration_asset(mismatched=True)
        with self.assertRaisesRegex(ValueError, "同形"):
            read_task(bad, bad["tasks"][0])

        benchmark, target = self.registration_asset()
        data = read_task(benchmark, benchmark["tasks"][0])
        shape = data["fixed_volumes"].shape
        ids = data["sample_ids"]
        base = {"ids": ids, "pred": target}
        for entry in (
            {**base, "registered": np.zeros((2, 5, 6, 6), dtype=np.float32)},
            {**base, "registered": np.full(shape, np.nan, dtype=np.float32)},
            {**base, "warped_prediction": np.zeros((2, 5, 6, 6), dtype=np.uint16)},
            {**base, "warped_prediction": np.full(shape, -1, dtype=np.int16)},
        ):
            with self.subTest(keys=set(entry)), self.assertRaises(ValueError):
                align_registration_volumes(entry, ids, shape)
        payload = self.registration_payload(benchmark, target)
        with self.assertRaises(ValueError):
            inspect_upload("prediction.npz", payload, "predictions", allow_registration_volumes=False)

    def test_legacy_two_dimensional_previews_remain_readable(self):
        folder = Path(self.temp.name) / "legacy"
        visuals = folder / "visuals"
        visuals.mkdir(parents=True)
        np.savez(visuals / "seg.npz", original=np.zeros((4, 5, 3), dtype=np.uint8),
                 overlay=np.ones((4, 5, 3), dtype=np.uint8))
        np.savez(visuals / "reg.npz", moving=np.zeros((3, 3), dtype=np.float32),
                 prediction=np.ones((3, 3), dtype=np.float32))
        self.assertEqual(set(load_preview(folder, {"kind": "segmentation", "file": "visuals/seg.npz"})), {"original", "overlay"})
        self.assertEqual(set(load_preview(folder, {"kind": "registration", "file": "visuals/reg.npz"})), {"moving", "prediction"})


if __name__ == "__main__":
    unittest.main()
