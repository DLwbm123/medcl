"""Reports contain scores and public protocols, never labels or local asset paths."""

import csv
from html import escape
import io
import json
import math
import re

from medcl import EVALUATOR_VERSION, VIEWER_SCHEMA_VERSION


CONTINUAL_METRICS = ("Final average", "BWT", "Forgetting", "FWT", "BWTR")
PUBLIC_WARNINGS = {
    "合成工程验收样例，禁止作为科研结果引用。",
    "未训练工程基线；不得作为方法性能或科研结论。",
    "已训练模型来源为提交者声明；平台未验证训练过程。",
    "外部预测来源未知；平台只验证测试评分，未验证训练过程。",
    "所有缺少阶段、参照或测试样本的指标保持不可计算；不插值、不补零。",
    "没有随机初始化 / 独立训练参照，FWT / RMA 不可计算。",
    "单机固定逻辑客户端评测模拟；仅聚合评分，没有联邦训练、权重聚合或通信成本。",
}


def submitter_result(result: dict) -> dict:
    """Remove classification query-oracle fields, including from legacy rows."""
    if not isinstance(result, dict):
        raise ValueError("评测结果格式无效")
    output = dict(result)
    kind = output.get("config", {}).get("benchmark", {}).get("kind")
    if kind == "classification":
        output.pop("cases", None)
        output["cells"] = [{key: value for key, value in item.items() if key != "n_cases"}
                           for item in output.get("cells", []) if isinstance(item, dict)]
        output["distributions"] = [
            {key: value for key, value in item.items() if key not in {"class_counts", "sample_ids", "case_id", "case_index"}}
            for item in output.get("distributions", []) if isinstance(item, dict)
        ]
    return output


def compatible_result(result: dict) -> dict:
    """Supply harmless display defaults for legacy results without changing scores."""
    output = submitter_result(result)
    config = output.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("benchmark"), dict):
        raise ValueError("评测结果缺少冻结配置")
    config = dict(config)
    output["config"] = config
    config.setdefault("evaluator_version", "legacy-unknown")
    config.setdefault("viewer_schema_version", "legacy-preview-unknown")
    if config["benchmark"].get("synthetic"):
        config["provenance"] = {"category": "synthetic", "statement": "由已校验的合成协议决定；仅用于工程验收",
                                "training_verified": False}
    elif not isinstance(config.get("provenance"), dict) or config["provenance"].get("category") not in {
            "untrained_baseline", "trained_model_declared", "external_predictions_unknown"}:
        config["provenance"] = {"category": "external_predictions_unknown",
                                "statement": "旧记录未保存有效结构化来源；按未知外部来源显示",
                                "training_verified": False}
    order = config.get("order", [])
    size = len(order)
    output.setdefault("result_schema_version", 1)
    for key in ("cells", "federated", "distributions", "visualizations", "warnings"):
        if not isinstance(output.get(key), list):
            output[key] = []
    if config["benchmark"].get("kind") != "classification" and not isinstance(output.get("cases"), list):
        output["cases"] = []
    matrices = output.get("matrices")
    if not isinstance(matrices, dict) or not isinstance(matrices.get("global"), list):
        matrices = {"global": [[None] * size for _ in range(size)]}
    output["matrices"] = matrices
    continual = output.get("continual")
    if not isinstance(continual, dict):
        continual = {}
    for client_id in matrices:
        summary = continual.get(client_id)
        if not isinstance(summary, dict):
            summary = {}
        continual[client_id] = {
            metric: summary.get(metric) if isinstance(summary.get(metric), dict) else
            {"value": None, "reason": "旧记录未保存该指标"}
            for metric in CONTINUAL_METRICS
        }
    output["continual"] = continual
    return output


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def report_cells(result: dict):
    result = compatible_result(result)
    config, b = result["config"], result["config"]["benchmark"]
    measured = {(c["stage"], c["task_id"], c["client_id"]): c for c in result["cells"]}
    for stage in range(1, len(config["order"]) + 1):
        for task_id in config["order"]:
            for client_id in result["matrices"]:
                yield measured.get((stage, task_id, client_id), {
                    "stage": stage, "task_id": task_id, "client_id": client_id, "score": None,
                    "metric": b["metric"], "direction": b["direction"], "unit": b["unit"],
                    "reason": "未提交该阶段" if stage not in config["stages"] else "协议不评测此任务",
                })


def report_csv(result: dict) -> bytes:
    result = compatible_result(result)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    fields = ("stage", "task_id", "client_id", "score", "metric", "direction", "unit", "n_samples", "n_cases", "reason", "benchmark_mean", "background")
    writer.writerow(["method", "benchmark", "version", "provenance_category", "training_verified", "synthetic",
                     "evaluator_version", "viewer_schema_version", "training_supervision", *fields])
    config, benchmark = result["config"], result["config"]["benchmark"]
    provenance = config["provenance"]
    for cell in report_cells(result):
        writer.writerow([_csv_value(v) for v in [config["method"], benchmark["id"], benchmark["version"],
                         provenance["category"], provenance.get("training_verified", False), benchmark["synthetic"],
                         config.get("evaluator_version", EVALUATOR_VERSION),
                         config.get("viewer_schema_version", VIEWER_SCHEMA_VERSION),
                         config.get("training_supervision", "not-declared"), *[cell.get(k) for k in fields]]])
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def report_json(result: dict) -> bytes:
    payload = compatible_result(result) if "config" in result else result
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")


def _value(value):
    return "不可计算" if value is None else f"{value:.6f}"


def report_html(result: dict) -> bytes:
    result = compatible_result(result)
    config = result["config"]
    b = config["benchmark"]
    task_names = {t["id"]: t["name"] for t in b["tasks"]}
    heading = " · 合成验收，非科研结果" if b["synthetic"] else " · 本轮实际评分"
    table = "".join("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in
                   (c["stage"], c["task_id"], c["client_id"], _value(c["score"]), c.get("n_samples", "—"), c["reason"])) + "</tr>" for c in report_cells(result))
    history = "".join(f"<tr><td>{escape(k)}</td><td>{_value(v['value'])}</td><td>{escape(v['reason'])}</td></tr>"
                      for k, v in result["continual"]["global"].items())
    client_rows = "".join("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in
                         (r["stage"], r["task_id"], _value(r["client_macro"]), _value(r["sample_weighted_accuracy"]),
                          _value(r["worst_client"]), _value(r["client_std"]), f"{r['available_clients']}/{r['total_clients']}")) + "</tr>" for r in result["federated"])
    warning_items = "".join(f"<li>{escape(w)}</li>" for w in result["warnings"])
    protocol = escape(json.dumps(config, ensure_ascii=False, indent=2))
    supervision = {"full": "全监督", "weak": "弱监督", "not-declared": "未声明"}.get(config.get("training_supervision"), "未声明")
    supervision_row = f"<p>分割训练监督：{supervision}（提交者声明，平台仅验证测试分数）</p>" if b["kind"] == "segmentation" else ""
    provenance = ({"category": "synthetic", "statement": "由已校验的合成协议决定；仅用于工程验收"}
                  if b.get("synthetic") else config.get("provenance", {
                      "category": "external_predictions_unknown", "statement": "旧记录未保存结构化来源；按未知外部来源显示"}))
    provenance_row = f"<p>结果来源：{escape(str(provenance.get('category')))} · {escape(str(provenance.get('statement')))}</p>"
    detail_note = "分类报告不包含逐样本正误、样本 ID 或隐藏标签频数。" if b["kind"] == "classification" else "病例明细见 JSON；隐藏测试真值不导出。"
    html = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
    <title>MedCL · {escape(config['method'])}</title><style>
    body{{font:16px/1.7 -apple-system,BlinkMacSystemFont,'PingFang SC','Microsoft YaHei',sans-serif;color:#243746;background:#f7f9fa;max-width:1100px;margin:40px auto;padding:0 24px}}
    h1,h2{{font-weight:600}} h2{{margin-top:32px}}table{{border-collapse:collapse;width:100%;background:white;font-size:14px}}td,th{{padding:10px 12px;border:1px solid #d9e1e6;text-align:left}}th{{background:#edf1f3}}pre{{white-space:pre-wrap;word-break:break-word;font-size:13px}}.notice{{padding:14px;border-left:4px solid #658776;background:#eaf0ed}}@media print{{body{{margin:0;max-width:none}}tr{{break-inside:avoid}}}}
    </style></head><body><p>MedCL / 医学影像持续学习评测</p><h1>{escape(config['method'])}</h1>
    <p>{escape(b['title'])}{heading} · {escape(b['version'])}</p>
    <p>任务顺序：{' → '.join(escape(t + ' ' + task_names[t]) for t in config['order'])}</p>
    {supervision_row}{provenance_row}
    <p>主指标：{escape(b['metric'])} · {'越低越好' if b['direction']=='lower' else '越高越好'} · {escape(b['unit'])}</p>
    <p>{escape(config['conditions'])}</p><div class="notice"><ul>{warning_items}</ul></div>
    <h2>阶段 × 客户端 × 测试任务</h2><table><thead><tr><th>阶段</th><th>任务</th><th>客户端</th><th>分数</th><th>样本数</th><th>状态</th></tr></thead><tbody>{table}</tbody></table>
    <h2>持续学习指标</h2><table><tr><th>指标</th><th>值</th><th>条件 / 原因</th></tr>{history}</table>
    <h2>固定逻辑客户端统计</h2><p>{escape(config['client_split']['source'])}</p><table><tr><th>阶段</th><th>任务</th><th>客户端宏平均</th><th>样本加权准确率</th><th>最低性能客户端</th><th>标准差</th><th>有样本客户端</th></tr>{client_rows}</table>
    <h2>冻结配置与来源</h2><pre>{protocol}</pre><p>原始标签、影像、私有权重及服务器路径不包含在本报告中。{detail_note}</p></body></html>'''
    return html.encode("utf-8")


def compatibility(config: dict) -> str:
    fields = {k: config[k] for k in ("benchmark", "order", "clients", "client_split", "evaluate_unseen", "output_head", "conditions")}
    fields["evaluator_version"] = config.get("evaluator_version", "legacy-unknown")
    fields["training_supervision"] = config.get("training_supervision", "not-declared")
    fields["test_assets"] = config.get("test_assets", [])
    return json.dumps(fields, sort_keys=True, ensure_ascii=False)


def aggregate_report(results: list[dict]) -> bytes:
    """Build a publication candidate from an explicit allow-list, never full config."""
    def label(value, fallback):
        if isinstance(value, str) and 1 <= len(value) <= 80 and all(c.isalnum() or c in " -_·（）()" for c in value):
            return value
        return fallback

    def number(value):
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("公开聚合只接受有限数值或 null")
        return value

    def version(value):
        return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", value) else "legacy-unknown"

    runs = []
    for raw in results:
        result = compatible_result(raw)
        config, benchmark = result["config"], result["config"]["benchmark"]
        job_id = result.get("job_id", "")
        safe_job = job_id if isinstance(job_id, str) and re.fullmatch(r"[a-f0-9]{32}", job_id) else "unknown"
        provenance = config.get("provenance", {})
        category = provenance.get("category") if isinstance(provenance, dict) else None
        if benchmark.get("synthetic") is True:
            category = "synthetic"
        elif category not in {"untrained_baseline", "trained_model_declared", "external_predictions_unknown"}:
            category = "external_predictions_unknown"
        metric = {"classification": ("Accuracy", "higher", "fraction"),
                  "segmentation": ("Foreground Dice", "higher", "fraction"),
                  "registration": ("TRE", "lower", "mm")}.get(benchmark.get("kind"))
        if metric is None or tuple(benchmark.get(key) for key in ("metric", "direction", "unit")) != metric:
            raise ValueError("公开聚合的指标协议无效")
        cells = []
        for cell in result["cells"]:
            if cell.get("client_id") != "global":
                continue
            if type(cell.get("stage")) is not int or type(cell.get("n_samples")) is not int or cell["n_samples"] < 0:
                raise ValueError("公开聚合单元格计数无效")
            safe_cell = {"stage": cell["stage"], "task_id": label(cell.get("task_id"), "invalid-task"),
                         "score": number(cell.get("score")), "n_samples": cell["n_samples"],
                         "metric": metric[0], "direction": metric[1], "unit": metric[2]}
            if "n_cases" in cell:
                if type(cell["n_cases"]) is not int or cell["n_cases"] < 0:
                    raise ValueError("公开聚合病例计数无效")
                safe_cell["n_cases"] = cell["n_cases"]
            for key in ("benchmark_mean", "background"):
                if key in cell:
                    safe_cell[key] = number(cell[key])
            cells.append(safe_cell)
        matrix = [[number(value) for value in row] for row in result["matrices"]["global"]]
        runs.append({
            "run_id": f"run-{safe_job[:8]}",
            "evaluator_version": version(config.get("evaluator_version")),
            "benchmark": {
                "id": label(benchmark.get("id"), "unknown-benchmark"),
                "public_title": label(benchmark.get("public_title", benchmark.get("title")), label(benchmark.get("id"), "MedCL benchmark")),
                "version": label(benchmark.get("version"), "unpublished-version"),
                "synthetic": benchmark.get("synthetic") is True,
                "metric": metric[0], "direction": metric[1], "unit": metric[2],
                "task_order": [label(task, "invalid-task") for task in config.get("order", [])],
            },
            "provenance": {"category": category, "training_verified": False},
            "global_cells": cells,
            "global_matrix": matrix,
            "global_continual": {name: {"value": number(result["continual"]["global"][name].get("value"))}
                                  for name in CONTINUAL_METRICS},
            "warnings": [warning for warning in result["warnings"] if warning in PUBLIC_WARNINGS],
        })
    return report_json({"schema": "medcl.aggregate-report.v2", "runs": runs,
                        "note": "显式白名单生成的全局聚合发布候选；不含方法自由文本、病例/客户端明细、标签、预测、模型、资产元数据或管理员私有说明，发布前仍需人工复核。"})


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from medcl.storage import get_job
    parser = argparse.ArgumentParser(description="Export explicitly selected completed runs as aggregate-only JSON")
    parser.add_argument("job_ids", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jobs = [get_job(jid) for jid in args.job_ids]
    if any(job is None or job["status"] != "completed" for job in jobs):
        raise SystemExit("只可导出已完成的评测；不会把失败记录称为结果")
    with args.output.open("xb") as handle:
        handle.write(aggregate_report([job["result"] for job in jobs]))
    print(f"Aggregate report saved: {args.output}")
