"""Explicit, offline import from the author's ZIP; never run during app startup.

Requires Poppler (pdftotext/pdftoppm). Only the five allowlisted members are read.
The review seed is a cross-check, not the parser's data source.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile

TEX = "chapters/ch03_medcl_benchmark.tex"
PDFS = ["SAM_confusion_matrix", "task_robustness_adice",
        "task_robustness_bwt_from_bottom", "domain_memory_size_adice"]
SOURCES = [TEX] + [f"figures/ch03/{name}.pdf" for name in PDFS]
SCENARIOS = {"domain": "Domain-CL", "class": "Class-CL", "organ": "Organ-CL"}
GROUPS = {"正则化方法": "regularization", "回放方法": "replay",
          "参数隔离方法": "parameter_isolation", "类别增量专项方法": "class_specific",
          "参照方法": "reference"}


def parse_tables(source):
    rows, audit = [], []
    scenario = label = family = None
    for number, line in enumerate(source.splitlines(), 1):
        label_match = re.search(r"\\label\{(tab:benchmark-(domain|class|organ)-results)\}", line)
        if label_match:
            label, key = label_match.groups()
            scenario = SCENARIOS[key]
        if not scenario:
            continue
        if line.startswith("方法 &"):
            columns = [cell.strip() for cell in line.rstrip("\\ ").split("&")][1:]
        for text, group in GROUPS.items():
            if "textit{" + text + "}" in line:
                family = group
        if not re.match(r"^(Regu-|Repl-|ParIso-|ClassCL-|Non-CL |JointTrain )", line):
            continue
        cells = [cell.strip() for cell in line.rstrip("\\ ").split("&")]
        if len(cells) != len(columns) + 1:
            raise ValueError(f"Malformed table row {number}")
        method = cells[0]
        record = dict(scenario=scenario, study_id=f"ch03-main-{key}", method_id=method,
                      family=family, role="reference" if family == "reference" else "continual_method",
                      source_type="thesis_reference", platform_recomputed=False,
                      source_file=TEX, source_label=label, source_line=number, source_page=None)
        metrics = {}
        for metric, raw in zip(columns, cells[1:]):
            nums = re.findall(r"-?(?:\d+\.\d+|\.\d+|\d+)", raw)
            mean = float(nums[0]) if nums else None
            sd = float(nums[1]) if "\\pm" in raw else None
            metrics[metric] = dict(mean=mean, sd=sd,
                status="reported" if nums else "not_applicable_or_not_reported",
                missing_reason=None if nums else "原表 --：不适用或未报告，未进一步区分。",
                unit="Dice" if metric in ("A-Dice", "WCD", "E-FWT") else "ratio",
                direction="minimize" if metric in ("MPE", "DRR") else "maximize",
                uncertainty_kind="reported_sd" if sd is not None else None,
                uncertainty_scope="not_specified_in_main_table" if sd is not None else None,
                n_repeats=None, source_raw=raw,
                source_precision={"mean": len(nums[0].split(".")[1]) if nums and "." in nums[0] else (0 if nums else None),
                                  "sd": len(nums[1].split(".")[1]) if sd is not None and "." in nums[1] else (0 if sd is not None else None)})
        record["metrics"] = metrics
        record["drr_star"] = "*" in metrics["DRR"]["source_raw"]
        record["raw_image_replay"] = None if method == "JointTrain" else record["drr_star"]
        record["data_access"] = ("joint_full_history" if method == "JointTrain" else
                                 "historical_subspace" if method == "Repl-GPM" else
                                 "raw_image_replay" if record["drr_star"] else "no_raw_image_replay")
        rows.append(record)
        audit.append(dict(scenario=scenario, source_line=number, source_label=label, source_text=line))
    if [sum(r["scenario"] == s for r in rows) for s in SCENARIOS.values()] != [15, 17, 14]:
        raise ValueError("Expected 15/17/14 source rows")
    if len({(r["scenario"], r["method_id"]) for r in rows}) != 46:
        raise ValueError("Duplicate method/scenario")
    return rows, audit


def pdf_words(path):
    raw = subprocess.check_output(["pdftotext", "-bbox", str(path), "-"])
    return [(e.text, float(e.attrib["xMin"]), float(e.attrib["yMin"]))
            for e in ET.fromstring(raw).iter() if e.tag.endswith("}word")]


def import_reference(zip_path, seed_path, output):
    seed = json.loads(seed_path.read_text())
    if zip_path.stat().st_size > 100_000_000:
        raise ValueError("ZIP exceeds the 100 MB import limit")
    zip_sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    if zip_sha != seed["source_zip_sha256"]:
        raise ValueError("ZIP version differs from this handoff; review before importing")
    order = []
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        payload = {}
        for name in SOURCES:
            if names.count(name) != 1 or archive.getinfo(name).file_size > 2_000_000:
                raise ValueError(f"Missing, duplicate or oversized source: {name}")
            payload[name] = archive.read(name)
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()}
    if hashes != seed["source_hashes"]:
        raise ValueError("Source version differs from review seed; review the new source explicitly")
    rows, audit = parse_tables(payload[TEX].decode("utf-8"))
    seed_rows = {(r["scenario"], r["method_id"]): r for r in seed["main_results"]}
    for row in rows:
        expected = seed_rows[(row["scenario"], row["method_id"])]
        for field in ("family", "role", "raw_image_replay", "source_line", "source_label"):
            if row[field] != expected[field]:
                raise ValueError(f"Seed mismatch {row['method_id']} {field}")
        for metric, values in row["metrics"].items():
            for field in ("mean", "sd", "status", "source_raw"):
                if values[field] != expected["metrics"][metric][field]:
                    raise ValueError(f"Seed mismatch {row['scenario']} {row['method_id']} {metric} {field}")
        row["source_sha256"] = hashes[TEX]

    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="medcl-ch03-import-") as temp:
        for name in PDFS:
            path = Path(temp) / f"{name}.pdf"
            path.write_bytes(payload[f"figures/ch03/{name}.pdf"])
            if name == "SAM_confusion_matrix":
                text = subprocess.check_output(["pdftotext", "-layout", str(path), "-"]).decode()
                lines = [re.findall(r"0\.\d{3}", line) for line in text.splitlines()]
                matrix_rows = [list(map(float, line)) for line in lines if len(line) == 12]
                if len(matrix_rows) != 6:
                    raise ValueError("Expected six rows of twelve SAM annotations")
                sam = dict(seed["sam_matrices"])
                sam["SAM"] = [r[:6] for r in matrix_rows]
                sam["SAM-LoRA"] = [r[6:] for r in matrix_rows]
                for mode in ("SAM", "SAM-LoRA"):
                    if sam[mode] != seed["sam_matrices"][mode]:
                        raise ValueError(f"Matrix mismatch {mode}")
                sam.update(study_id="ch03-sam", source_sha256=hashes[sam["source_file"]])
            if name.startswith("task_robustness"):
                words = pdf_words(path)
                values = sorted([(x, float(text)) for text, x, y in words if re.fullmatch(r"0\.\d{4}", text)])
                methods = sorted([(x, text.replace("LWF", "LwF")) for text, x, y in words if text.startswith(("Regu-", "Repl-"))])
                if len(values) != 8 or len(methods) != 8:
                    raise ValueError("Expected eight SD annotations and method labels")
                metric = "A-Dice" if name.endswith("adice") else "BWTR"
                for (_, method), (_, sd) in zip(methods, values):
                    expected = next(r for r in seed["order_robustness"] if r["method_id"] == method and r["metric"] == metric)
                    if sd != expected["order_sd"]:
                        raise ValueError(f"Order SD mismatch {method} {metric}")
                    order.append(dict(expected, order_sd=sd, study_id="ch03-order", mean=None, sd=sd,
                                      family="regularization" if method.startswith("Regu-") else "replay",
                                      role="supplementary_study", source_line=None, source_precision=4,
                                      status="reported", unit="Dice" if metric == "A-Dice" else "ratio",
                                      direction="minimize", missing_reason="十种顺序的逐次精确观测未提供；仅导入原图标注 SD。",
                                      uncertainty_kind="sd_across_task_orders", platform_recomputed=False,
                                      source_sha256=hashes[expected["source_file"]]))
            (output / path.name).write_bytes(path.read_bytes())
            subprocess.run(["pdftoppm", "-scale-to", "1700", "-png", "-singlefile", str(path), str(output / name)], check=True)

    registry = dict(schema="medcl.ch03.reference.v1", source_type="thesis_reference", platform_recomputed=False,
                    source_zip_sha256=zip_sha,
                    source_hashes=hashes, main_results=rows, sam_matrices=sam, order_robustness=order,
                    memory_capacity_study=dict(seed["memory_capacity_study"], study_id="ch03-memory",
                        source_type="thesis_reference", platform_recomputed=False, source_page=1,
                        source_sha256=hashes[seed["memory_capacity_study"]["source_file"]]),
                    source_table_mapping=audit)
    (output / "results.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"Verified {len(rows)} rows / {sum(len(r['metrics']) for r in rows)} cells; 72 SAM values; {len(order)} order SD annotations. No capacity points imported.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", type=Path)
    parser.add_argument("seed", type=Path)
    parser.add_argument("--output", type=Path, default=Path("medcl/ui_assets/ch03"))
    args = parser.parse_args()
    import_reference(args.zip, args.seed, args.output)
