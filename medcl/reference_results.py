"""Read-only Chapter 3 history. This module has no job/storage/worker dependency."""
from collections import Counter
from functools import lru_cache
import json
import math
from pathlib import Path

ASSETS = Path(__file__).with_name("ui_assets") / "ch03"
NOTICE = "论文历史结果 · 第三章 · 非平台重新评测"
FAMILIES = {"regularization": "正则化", "replay": "回放", "parameter_isolation": "参数隔离",
            "class_specific": "专项", "reference": "参照"}
COLORS = dict(zip(FAMILIES, ["#356EB5", "#178477", "#815CB7", "#C97724", "#788595"]))
SCENARIOS = {"Domain-CL": "域增量 Domain-CL", "Class-CL": "类别增量 Class-CL", "Organ-CL": "任务增量 Organ-CL"}
TIMELINES = {"Domain-CL": "前列腺 MRI · A → B → C → D → E → F · 共享二通道头",
             "Class-CL": "心脏 CT · LV / LA / MYO → RV / RA → AA / PA · 七结构，三阶段，共享八通道头",
             "Organ-CL": "左心房 MRI → 前列腺 MRI → 肝 CT → 脑肿瘤 FLAIR MRI · 任务特定二通道头；跨器官，原论文标识 Organ-CL"}
METRICS = {
    "A-Dice": "最终阶段（矩阵最后一行）的任务平均 Dice；不是整个矩阵平均。越大越好。",
    "BWTR": "旧任务最终相对初学性能的变化率均值；不是普通 BWT。零均值可能抵消正负变化，不等于每个旧任务都没有遗忘。越大越好。",
    "RMA": "第 2 至 T 阶段初学 Dice / 各任务独立训练 Dice 的均值。分母不是主表 Non-CL 顺序训练结果，可大于 1。",
    "E-FWT": "仅 Domain-CL：未来域直接测试 Dice 减去随机初始化参照，再对严格上三角取平均。不是上三角 Dice 本身。",
    "WCD": "仅 Class-CL：最终统一输出空间的全类别 Dice。与 A-Dice 的汇总对象不同，差值不是时间变化或遗忘量。",
    "MPE": "新增参数相对初始参数量的阶段平均比例，越小扩展越少，可大于 1；不是训练时间、显存或模型字节数。",
    "DRR": "历史来源数据的回访量 / 对应训练数据量的平均比例。星号表示原始图像回放；GPM 为其他历史表示。不是通信或显存成本。",
}
PROTOCOL = {
    "source_file": "chapters/ch03_medcl_benchmark.tex", "source_lines": "370–641",
    "backbone": "ResUNet32（主表）；SAM 扩展独立展示",
    "split": "按病例 60% 训练 / 15% 验证 / 25% 测试；方法间使用相同划分，超参数基于验证集。",
    "training": "150 epochs；SGD；初始学习率 0.008，80 epoch 后乘 0.5；batch size 8。Domain/Organ 使用交叉熵，Class 使用 MiB 无偏交叉熵。",
    "replay": "默认容量 32，回放 batch size 8，reservoir sampling。",
    "Domain-CL": "T2 MRI：A/B 来自 NCI-ISBI2013，C 来自 I2CVB，D/E/F 来自 PROMISE12。C 裁剪对齐；轴向 256×256，非零区域均值/标准差归一化。",
    "Class-CL": "MMWHS CT：0.78×0.78 mm 平面分辨率，平均层厚 1.60 mm，256×256；七个心脏结构逐步加入。",
    "Organ-CL": "LAScarQS 左心房 LGE MRI → PROMISE12 中心 D → LiTS → FeTS2021 FLAIR；均值/标准差归一化，256×256。",
    "uncertainty": "主表只报告均值 ± SD；重复次数、SD 的变异来源及更细的病例聚合方式未充分明确，均保留未知。",
    "references": "Non-CL 是顺序训练参照。JointTrain 访问全部任务数据，其 DRR=0 不等于没有历史数据访问；两者不参加 CL 冠军或资源前沿。",
}


def validate(data):
    if data["schema"] != "medcl.ch03.reference.v1" or data["platform_recomputed"] is not False:
        raise ValueError("历史结果 schema / 来源无效")
    rows = data["main_results"]
    if Counter(r["scenario"] for r in rows) != {"Domain-CL": 15, "Class-CL": 17, "Organ-CL": 14}:
        raise ValueError("第三章主表记录不完整")
    if len({(r["scenario"], r["method_id"]) for r in rows}) != 46:
        raise ValueError("第三章主表记录重复")
    for r in rows:
        if r["source_type"] != "thesis_reference" or r["platform_recomputed"] is not False:
            raise ValueError("历史来源不允许混入实测记录")
        if r["family"] not in FAMILIES or r["source_sha256"] != data["source_hashes"][r["source_file"]]:
            raise ValueError("分类或来源不一致")
        for metric, v in r["metrics"].items():
            for field in ("mean", "sd"):
                if v[field] is not None and not math.isfinite(v[field]):
                    raise ValueError("不允许 NaN / Infinity")
            if v["sd"] is not None and (v["sd"] < 0 or v["mean"] is None):
                raise ValueError("SD 无效")
            if (v["status"] == "reported") != (v["mean"] is not None):
                raise ValueError("数值和缺失状态不一致")
            if metric in ("MPE", "DRR") and v["sd"] is not None:
                raise ValueError("资源指标没有报告 SD")
    sam = data["sam_matrices"]
    for mode in ("SAM", "SAM-LoRA"):
        if len(sam[mode]) != 6 or any(len(row) != 6 for row in sam[mode]):
            raise ValueError("SAM 矩阵应为 6×6")
        if any(not 0 <= v <= 1 for row in sam[mode] for v in row):
            raise ValueError("SAM Dice 无效")
    if len(data["order_robustness"]) != 16 or data["memory_capacity_study"]["exact_numeric_series_available_in_zip"]:
        raise ValueError("补充研究数据边界不符")
    return data


@lru_cache(maxsize=1)
def load_reference():
    return validate(json.loads((ASSETS / "results.json").read_text()))


def select_rows(rows, scenario=None, families=None, methods=None, references=False):
    return [r for r in rows if (scenario is None or r["scenario"] == scenario)
            and (families is None or r["family"] in families or r["role"] == "reference")
            and (methods is None or r["method_id"] in methods)
            and (references or r["role"] != "reference")]


def value(row, metric):
    return row["metrics"].get(metric, {}).get("mean")


def winner(rows, metric="A-Dice", fixed=False):
    candidates = [r for r in rows if r["role"] == "continual_method" and value(r, metric) is not None
                  and (not fixed or value(r, "MPE") == 0)]
    if not candidates:
        return None, []
    best = max(value(r, metric) for r in candidates)
    return best, [r["method_id"] for r in candidates if value(r, metric) == best]


def pareto(rows, resource):
    candidates = [r for r in rows if r["role"] == "continual_method"
                  and value(r, resource) is not None and value(r, "A-Dice") is not None]
    return [r["method_id"] for r in candidates if not any(
        value(s, resource) <= value(r, resource) and value(s, "A-Dice") >= value(r, "A-Dice")
        and (value(s, resource) < value(r, resource) or value(s, "A-Dice") > value(r, "A-Dice")) for s in candidates)]


def flatten(rows):
    output = []
    for row in rows:
        source = {k: v for k, v in row.items() if k != "metrics"}
        for metric in METRICS:
            cell = row["metrics"].get(metric, dict(mean=None, sd=None, status="not_applicable",
                missing_reason=f"{metric} 不适用于 {row['scenario']}，不属于原表列。", unit="Dice",
                direction="maximize", source_precision={"mean": None, "sd": None}, source_raw=None,
                uncertainty_kind=None, uncertainty_scope=None, n_repeats=None))
            output.append(dict(source, metric=metric, **cell))
    return output


def sam_cells(data):
    sam = data["sam_matrices"]
    source = {k: sam[k] for k in ("source_file", "source_label", "source_page", "source_sha256", "study_id", "source_type", "platform_recomputed")}
    return [dict(source, scenario="Domain-CL", family="sam_adaptation", role="supplementary_study", source_line=None,
                 method_id=mode, stage=i + 1, test_task=f"T{j + 1}", metric="Dice", mean=v, sd=None,
                 unit="Dice", direction="maximize", status="reported", missing_reason=None, source_precision=3,
                 region="当前任务" if i == j else "历史任务" if i > j else "未来域直接测试")
            for mode in ("SAM", "SAM-LoRA") for i, row in enumerate(sam[mode]) for j, v in enumerate(row)]


def sam_summary(data):
    result = {}
    for mode in ("SAM", "SAM-LoRA"):
        matrix = data["sam_matrices"][mode]
        result[mode] = {"final_A_Dice": sum(matrix[-1]) / 6,
                        "matrix_mean": sum(map(sum, matrix)) / 36,
                        "off_diagonal_mean": sum(matrix[i][j] for i in range(6) for j in range(6) if i != j) / 30}
    return result


def cross_records(rows, methods):
    """Export absent method/scenario pairs as missing observations, never as zero."""
    result = flatten(rows)
    observed = {(r["method_id"], r["scenario"]) for r in rows}
    for method in methods:
        for scenario in SCENARIOS:
            if (method, scenario) not in observed:
                sample = next(r for r in rows if r["method_id"] == method)
                label = f"tab:benchmark-{scenario.split('-')[0].lower()}-results"
                for metric in ("A-Dice", "BWTR", "RMA"):
                    result.append(dict(scenario=scenario, study_id=f"ch03-main-{scenario.split('-')[0].lower()}",
                        method_id=method, family=sample["family"], role=sample["role"], metric=metric, mean=None, sd=None,
                        unit="Dice" if metric == "A-Dice" else "ratio", direction="maximize", status="not_reported",
                        missing_reason="该场景主表没有报告此方法；不代表不适用或得分为零。",
                        source_file=sample["source_file"], source_label=label, source_line=None, source_page=None,
                        source_sha256=sample["source_sha256"], source_type="thesis_reference", platform_recomputed=False,
                        source_precision=None))
    return result
