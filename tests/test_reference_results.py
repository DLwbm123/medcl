"""Independent paper anchors plus every-cell mapping back to the source table text."""
from collections import Counter
import copy
import re
import unittest

from medcl.reference_results import *

SAM = [[.888,.811,.790,.728,.776,.580], [.784,.898,.752,.663,.706,.542],
       [.714,.610,.907,.887,.781,.793], [.733,.783,.859,.875,.695,.709],
       [.886,.842,.817,.826,.855,.666], [.496,.496,.697,.632,.518,.887]]
LORA = [[.877,.841,.839,.820,.844,.702], [.915,.913,.863,.856,.869,.708],
        [.886,.886,.916,.877,.845,.799], [.914,.900,.917,.915,.829,.818],
        [.861,.753,.835,.829,.867,.721], [.760,.534,.718,.670,.627,.894]]


class ReferenceResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_reference()
        cls.rows = cls.data["main_results"]

    def test_all_46_rows_262_cells_against_source_text(self):
        mapping = self.data["source_table_mapping"]
        self.assertEqual(len(mapping), 46)
        seen = 0
        for row, original in zip(self.rows, mapping):
            self.assertEqual(row["source_line"], original["source_line"])
            self.assertEqual(row["source_label"], original["source_label"])
            cells = original["source_text"].split("&")
            self.assertEqual(row["method_id"], cells[0].strip())
            self.assertEqual(len(cells) - 1, len(row["metrics"]))
            for v, raw in zip(row["metrics"].values(), cells[1:]):
                # Different numeric path from the importer; keep whole source row reviewable.
                clean = re.sub(r"[^0-9.\-;]", "", raw.replace("\\pm", ";"))
                if clean == "--":
                    self.assertIsNone(v["mean"])
                    self.assertIsNone(v["sd"])
                    self.assertIsNotNone(v["missing_reason"])
                else:
                    parts = clean.split(";")
                    self.assertEqual(v["mean"], float(parts[0]))
                    self.assertEqual(v["sd"], float(parts[1]) if len(parts) == 2 else None)
                seen += 1
            self.assertEqual(row["drr_star"], "*" in cells[-1])
        self.assertEqual(seen, 262)

    def test_composition_missing_zero_and_reference_boundaries(self):
        expected = {"Domain-CL": [3, 8, 2, 0], "Class-CL": [3, 8, 2, 2], "Organ-CL": [3, 7, 2, 0]}
        for scenario, counts in expected.items():
            rows = select_rows(self.rows, scenario)
            families = Counter(r["family"] for r in rows)
            self.assertEqual([families[f] for f in list(FAMILIES)[:-1]], counts)
        lookup = {(r["scenario"], r["method_id"]): r for r in self.rows}
        self.assertNotIn(("Organ-CL", "Repl-DER++"), lookup)
        pnn = lookup["Domain-CL", "ParIso-PNN"]
        self.assertEqual((pnn["metrics"]["BWTR"]["mean"], pnn["metrics"]["BWTR"]["sd"]), (0, 0))
        self.assertEqual(value(pnn, "MPE"), 1.112)
        self.assertIsNone(pnn["metrics"]["MPE"]["sd"])
        ewc = lookup["Domain-CL", "Regu-EWC"]
        self.assertEqual((value(ewc, "RMA"), value(ewc, "BWTR")), (1.086, -.242))
        for r in self.rows:
            if r["method_id"] == "Repl-GPM":
                self.assertEqual(r["family"], "replay")
                self.assertFalse(r["raw_image_replay"])
            if r["method_id"] == "JointTrain":
                self.assertEqual(r["data_access"], "joint_full_history")
        joint = lookup["Class-CL", "JointTrain"]
        self.assertIsNone(value(joint, "WCD"))
        rows = select_rows(self.rows, "Class-CL", references=True)
        self.assertEqual(winner(rows, fixed=True), (.751, ["ClassCL-MiB"]))
        self.assertEqual(winner(rows, "WCD"), (.707, ["ClassCL-PLOP"]))
        self.assertEqual(winner([], "BWTR"), (None, []))
        self.assertEqual(set(winner(rows, "BWTR")[1]), {"ParIso-PNN", "ParIso-DAN"})
        for resource in ("MPE", "DRR"):
            self.assertNotIn("JointTrain", pareto(rows, resource))
            self.assertNotIn("Non-CL", pareto(rows, resource))
        self.assertEqual(select_rows(self.rows, families=[]), [])

    def test_all_sam_annotations_orientation_and_derived_values(self):
        self.assertEqual(self.data["sam_matrices"]["SAM"], SAM)
        self.assertEqual(self.data["sam_matrices"]["SAM-LoRA"], LORA)
        self.assertEqual(len(sam_cells(self.data)), 72)
        stats = sam_summary(self.data)
        for mode, final, overall, off in [("SAM", .621, .7467222222, .7190666667), ("SAM-LoRA", .7005, .8227222222, .8078666667)]:
            self.assertAlmostEqual(stats[mode]["final_A_Dice"], final)
            self.assertAlmostEqual(stats[mode]["matrix_mean"], overall)
            self.assertAlmostEqual(stats[mode]["off_diagonal_mean"], off)
            self.assertNotIn("RMA", stats[mode])
            self.assertNotIn("E-FWT", stats[mode])
        deltas = [LORA[i][j] - SAM[i][j] for i in range(6) for j in range(6)]
        self.assertEqual((sum(d > 0 for d in deltas), sum(d < 0 for d in deltas)), (32, 4))
        cells = sam_cells(self.data)
        self.assertEqual(next(c for c in cells if c["method_id"] == "SAM" and c["stage"] == 1 and c["test_task"] == "T6")["mean"], .580)

    def test_order_sd_independent_from_main_sd_and_no_capacity_points(self):
        expected = {"Regu-EWC": (.0280,.0274), "Regu-LwF": (.0328,.0226), "Repl-ER": (.0204,.0145),
            "Repl-GSS": (.0134,.0123), "Repl-GEM": (.0238,.0189), "Repl-AGEM": (.0239,.0145), "Repl-DER": (.0141,.0190), "Repl-FDR": (.0206,.0136)}
        for record in self.data["order_robustness"]:
            self.assertEqual(record["order_sd"], expected[record["method_id"]][0 if record["metric"] == "A-Dice" else 1])
            self.assertEqual(record["n_orders"], 10)
            self.assertFalse(record["order_values_available"])
            self.assertIsNone(record["mean"])
        capacity = self.data["memory_capacity_study"]
        self.assertFalse(capacity["exact_numeric_series_available_in_zip"])
        self.assertFalse({"points", "series", "values"} & set(capacity))

    def test_schema_rejects_nonfinite_duplicate_and_mixed_sources(self):
        for mutate in [lambda d: d["main_results"].append(d["main_results"][0]),
                       lambda d: d["main_results"][0]["metrics"]["BWTR"].update(mean=float("nan")),
                       lambda d: d["main_results"][0].update(source_type="platform")]:
            altered = copy.deepcopy(self.data)
            mutate(altered)
            with self.assertRaises(ValueError): validate(altered)


if __name__ == "__main__":
    unittest.main()
