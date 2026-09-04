"""Reports contain scores and public protocols, never labels or local asset paths."""

import csv
from html import escape
import io
import json


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def report_cells(result: dict):
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
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    fields = ("stage", "task_id", "client_id", "score", "metric", "direction", "unit", "n_samples", "n_cases", "reason", "benchmark_mean", "background")
    writer.writerow(["method", "benchmark", "version", "synthetic", "training_supervision", *fields])
    config, benchmark = result["config"], result["config"]["benchmark"]
    for cell in report_cells(result):
        writer.writerow([_csv_value(v) for v in [config["method"], benchmark["id"], benchmark["version"], benchmark["synthetic"], config.get("training_supervision", "not-declared"), *[cell.get(k) for k in fields]]])
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def report_json(result: dict) -> bytes:
    return json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")


def _value(value):
    return "不可计算" if value is None else f"{value:.6f}"


def report_html(result: dict) -> bytes:
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
    html = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
    <title>MedCL · {escape(config['method'])}</title><style>
    body{{font:16px/1.7 -apple-system,BlinkMacSystemFont,'PingFang SC','Microsoft YaHei',sans-serif;color:#243746;background:#f7f9fa;max-width:1100px;margin:40px auto;padding:0 24px}}
    h1,h2{{font-weight:600}} h2{{margin-top:32px}}table{{border-collapse:collapse;width:100%;background:white;font-size:14px}}td,th{{padding:10px 12px;border:1px solid #d9e1e6;text-align:left}}th{{background:#edf1f3}}pre{{white-space:pre-wrap;word-break:break-word;font-size:13px}}.notice{{padding:14px;border-left:4px solid #658776;background:#eaf0ed}}@media print{{body{{margin:0;max-width:none}}tr{{break-inside:avoid}}}}
    </style></head><body><p>MedCL / 医学影像持续学习评测</p><h1>{escape(config['method'])}</h1>
    <p>{escape(b['title'])}{heading} · {escape(b['version'])}</p>
    <p>任务顺序：{' → '.join(escape(t + ' ' + task_names[t]) for t in config['order'])}</p>
    {supervision_row}
    <p>主指标：{escape(b['metric'])} · {'越低越好' if b['direction']=='lower' else '越高越好'} · {escape(b['unit'])}</p>
    <p>{escape(config['conditions'])}</p><div class="notice"><ul>{warning_items}</ul></div>
    <h2>阶段 × 客户端 × 测试任务</h2><table><thead><tr><th>阶段</th><th>任务</th><th>客户端</th><th>分数</th><th>样本数</th><th>状态</th></tr></thead><tbody>{table}</tbody></table>
    <h2>持续学习指标</h2><table><tr><th>指标</th><th>值</th><th>条件 / 原因</th></tr>{history}</table>
    <h2>固定逻辑客户端统计</h2><p>{escape(config['client_split']['source'])}</p><table><tr><th>阶段</th><th>任务</th><th>客户端宏平均</th><th>样本加权准确率</th><th>最低性能客户端</th><th>标准差</th><th>有样本客户端</th></tr>{client_rows}</table>
    <h2>冻结配置与来源</h2><pre>{protocol}</pre><p>原始标签、影像、私有权重及服务器路径不包含在本报告中。病例明细见 JSON。</p></body></html>'''
    return html.encode("utf-8")


def compatibility(config: dict) -> str:
    fields = {k: config[k] for k in ("benchmark", "order", "clients", "client_split", "evaluate_unseen", "output_head", "conditions")}
    fields["training_supervision"] = config.get("training_supervision", "not-declared")
    fields["test_assets"] = config.get("test_assets", [])
    return json.dumps(fields, sort_keys=True, ensure_ascii=False)


def aggregate_report(results: list[dict]) -> bytes:
    """Publication candidate: global aggregates only, never patient/client case-level records."""
    runs = [{"job_id": r["job_id"], "config": r["config"],
             "global_cells": [c for c in r["cells"] if c["client_id"] == "global"],
             "global_matrix": r["matrices"]["global"], "continual": r["continual"]["global"],
             "warnings": r["warnings"]} for r in results]
    return report_json({"schema": "medcl.aggregate-report.v1", "runs": runs,
                        "note": "仅全局聚合指标。不含病例明细、客户端个体结果、标签、原始预测、模型或资产路径。发布前仍需检查方法名称与管理员协议文字。"})


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
