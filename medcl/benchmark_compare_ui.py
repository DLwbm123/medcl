"""Chapter 3 comparison page. Historical inputs stay outside platform jobs."""
from collections import Counter
from html import escape
import json

import pandas as pd
import streamlit as st

from medcl import benchmark_charts as charts
from medcl.reference_exports import export_view
from medcl.reference_results import (ASSETS, COLORS, FAMILIES, METRICS, NOTICE, PROTOCOL,
    SCENARIOS, TIMELINES, cross_records, flatten, load_reference, sam_cells, sam_summary, select_rows, winner)

VIEWS = ["总览", "能力与资源", "跨场景", "鲁棒性", "SAM扩展", "数据与来源"]


def reset_filters():
    for key in list(st.session_state):
        if key.startswith("ch03-") and key not in ("ch03-scenario", "ch03-view"):
            del st.session_state[key]


def to_domain():
    st.session_state["ch03-scenario"] = "Domain-CL"
    reset_filters()


def picture(name, caption, pictures):
    path = ASSETS / f"{name}.png"
    st.image(str(path), caption=caption, width="stretch")
    st.download_button("查看 / 下载原论文 PDF", (ASSETS / f"{name}.pdf").read_bytes(),
                       file_name=f"{name}.pdf", mime="application/pdf", key=f"ch03-pdf-{name}")
    pictures.append((path, caption))


def kpis(rows, scenario):
    key = {"Domain-CL": "E-FWT", "Class-CL": "WCD", "Organ-CL": "BWTR"}[scenario]
    best, names = winner(rows)
    fixed, fixed_names = winner(rows, fixed=True)
    special, special_names = winner(rows, key)
    count = sum(r["role"] == "continual_method" for r in rows)
    items = [("CL 方法数", str(count), "当前筛选 · 不含参照"),
             ("最高 A-Dice", "—" if best is None else f"{best:.3f}", " / ".join(names) or "无可用结果"),
             ("固定参数 · 最高 A-Dice", "—" if fixed is None else f"{fixed:.3f}", " / ".join(fixed_names) or "无可用结果"),
             (f"最高 {key}", "—" if special is None else f"{special:.3f}", " / ".join(special_names) or "无可用结果")]
    st.html("<div class='ch03-kpis'>" + "".join(f"<div><span>{escape(label)}</span><strong>{escape(number)}</strong><small>{escape(note)}</small></div>" for label, number, note in items) + "</div>")


def render():
    try:
        registry = load_reference()
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"第三章历史数据加载失败：{exc}")
        return
    with st.container(key="ch03-benchmark"):
        st.html("""<style>
        .st-key-ch03-benchmark{min-width:0;gap:.55rem}
        .st-key-ch03-benchmark [data-testid="stVerticalBlock"]{gap:.55rem}
        .st-key-ch03-benchmark [data-testid="stMarkdownContainer"] p{margin-bottom:.35rem}
        .st-key-ch03-benchmark .ch03-kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:2px 0 16px}
        .ch03-kpis>div{background:white;border:1px solid #DDE5EE;border-radius:12px;padding:16px;min-width:0}
        .ch03-kpis span{font-size:14px;font-weight:650;color:#53677F}
        .ch03-kpis strong{display:block;font-size:28px;line-height:1.5;color:#355F8A}
        .ch03-kpis small{display:block;font-size:12px;color:#66768A;overflow-wrap:anywhere}
        .st-key-ch03-benchmark [data-testid="stButtonGroup"] [role="radiogroup"]{flex-wrap:wrap}
        .st-key-ch03-benchmark [data-testid="stVegaLiteChart"]{overflow:hidden;border-radius:10px;border:1px solid #E3EAF2;background:white}
        @media(max-width:700px){.st-key-ch03-benchmark .ch03-kpis{grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.ch03-kpis>div{padding:12px}.ch03-kpis strong{font-size:24px}}
        </style>""")
        st.subheader("持续分割 Benchmark")
        st.caption(f"{NOTICE} · 全监督 · ResUNet32 主表")
        if st.session_state.get("ch03-scenario") not in SCENARIOS:
            st.session_state["ch03-scenario"] = "Domain-CL"
        scenario = st.segmented_control("增量场景", list(SCENARIOS), format_func=SCENARIOS.get,
            key="ch03-scenario", on_change=reset_filters, label_visibility="collapsed") or "Domain-CL"
        st.caption(TIMELINES[scenario])
        view = st.segmented_control("第三章视图", VIEWS, default="总览", key="ch03-view", label_visibility="collapsed") or "总览"
        with st.expander("筛选与显示 · 默认展示全部已报告 CL 方法"):
            a, b = st.columns(2)
            families = a.multiselect("方法类别", list(FAMILIES)[:-1], default=list(FAMILIES)[:-1], format_func=FAMILIES.get, key="ch03-families")
            references = b.checkbox("显示 Non-CL / JointTrain 参照", key="ch03-references")
            sd = b.checkbox("显示论文 SD 误差线", value=True, key="ch03-sd")
            metrics = ["A-Dice", "BWTR", "RMA"] + (["E-FWT"] if scenario == "Domain-CL" else ["WCD"] if scenario == "Class-CL" else [])
            if st.session_state.get("ch03-metric", "A-Dice") not in metrics:
                st.session_state["ch03-metric"] = "A-Dice"
            metric = b.selectbox("主排名指标", metrics, key="ch03-metric")
            options = select_rows(registry["main_results"], None if view == "跨场景" else scenario, families, references=references)
            methods = list(dict.fromkeys(r["method_id"] for r in options))
            if "ch03-methods" in st.session_state:
                previous = st.session_state.get("ch03-method-options", methods)
                st.session_state["ch03-methods"] = [m for m in st.session_state["ch03-methods"] if m in methods] + [m for m in methods if m not in previous]
            else:
                st.session_state["ch03-methods"] = methods
            st.session_state["ch03-method-options"] = methods
            selected = st.multiselect("方法（可多选）", methods, key="ch03-methods")
            st.button("重置筛选", on_click=reset_filters, key="ch03-reset")
            st.caption("固定参数指主表 MPE=0。参照不参加 CL 冠军、方法构成或资源前沿。主表没有逐阶段轨迹和病例级分布。")
        rows = select_rows(registry["main_results"], scenario, families, selected, references)
        scope = dict(view=view, scenario=scenario, families=families, methods=selected, references=references, reported_sd=sd)
        records, figures, pictures = flatten(rows), {}, []

        def show(name, chart):
            figures[name] = chart
            st.altair_chart(chart, width="stretch", theme=None)

        if view in ("鲁棒性", "SAM扩展") and scenario != "Domain-CL":
            st.info("该补充研究仅报告 Domain-CL；没有当前场景的对应结果。")
            st.button("切换到 Domain-CL", on_click=to_domain, key="ch03-domain")
            return
        if view not in ("鲁棒性", "SAM扩展", "数据与来源") and not selected:
            kpis([], scenario)
            st.info("当前筛选没有方法。请添加方法或重置筛选。")
        elif view == "总览":
            kpis(rows, scenario)
            left, right = st.columns([2, 1], gap="large")
            with left:
                scope["metric"] = metric
                show(f"{scenario}-{metric}", charts.ranking(rows, metric, sd))
                st.caption(METRICS[metric])
                missing = [r["method_id"] for r in rows if r["metrics"].get(metric, {}).get("mean") is None]
                if missing:
                    st.caption("未报告 / 不适用：" + "、".join(missing) + "；保留缺失，不绘制为零。")
            with right:
                counts = Counter(r["family"] for r in rows if r["role"] == "continual_method")
                if counts:
                    show("方法构成", charts.composition(rows))
                    total = sum(counts.values())
                    for family, n in counts.items():
                        st.markdown(f"**{FAMILIES[family]}**　{n} 个 · {n / total:.1%}")
                st.caption("当前筛选的方法数量构成，不表示性能占比。")
                st.markdown("**如何阅读结果**")
                st.write("KPI 仅比较当前筛选中的 CL 报告均值；固定参数集合额外要求 MPE=0。均值差异不代表统计显著性。")
                st.caption("误差线为论文报告 SD；主表重复次数与变异来源未知。JointTrain 使用全部任务数据，仅作为参照。")
        elif view == "能力与资源":
            st.caption("每张图回答一个能力或资源问题；不计算跨指标综合冠军。")
            a, b = st.columns(2)
            with a: show("BWTR", charts.ranking(rows, "BWTR", sd))
            with b: show("RMA", charts.ranking(rows, "RMA", sd))
            st.caption(METRICS["BWTR"] + " " + METRICS["RMA"])
            a, b = st.columns(2)
            with a: show("稳定性与可塑性", charts.tradeoff(rows))
            with b:
                if scenario == "Domain-CL":
                    show("E-FWT", charts.ranking(rows, "E-FWT", sd))
                    st.caption(METRICS["E-FWT"])
                elif scenario == "Class-CL":
                    show("A-Dice-WCD", charts.dumbbell(rows, sd))
                    st.caption("MiB 的 A-Dice 高于 PLOP，PLOP 的 WCD 高于 MiB（原表结论）。两个指标汇总对象不同；JointTrain 未报告 WCD。")
            frontier = st.checkbox("标出当前 CL 集合内的二维 Pareto 前沿", value=True, key="ch03-pareto")
            scope["pareto"] = frontier
            a, b = st.columns(2)
            with a: show("MPE-A-Dice", charts.tradeoff(rows, "MPE", frontier))
            with b: show("DRR-A-Dice", charts.tradeoff(rows, "DRR", frontier))
            st.caption("黑色轮廓 / 虚线为按当前 CL 报告均值计算的二维非支配集合，不含参照；不表示统计显著性或相同全部资源预算。")
            st.caption(METRICS["MPE"] + " " + METRICS["DRR"] + " JointTrain 的 DRR=0 不代表没有历史数据访问。")
        elif view == "跨场景":
            metric = st.selectbox("共同指标", ["A-Dice", "BWTR", "RMA"], key="ch03-common-metric")
            scope["metric"] = metric
            cross_rows = select_rows(registry["main_results"], None, families, selected, references)
            records = cross_records(cross_rows, selected)
            show("跨场景热图", charts.cross_scene(cross_rows, selected, metric))
            st.caption("每列是独立场景，展示同一指标原始值，不构造三场景综合分。Organ-CL 没有报告 DER++；空白不是零。")
            st.caption(METRICS[metric])
        elif view == "鲁棒性":
            scope["filter_scope"] = "仅 Domain-CL 八种已报告方法；方法筛选作用于顺序 SD，容量原图为完整原文。"
            order = [r for r in registry["order_robustness"] if r["method_id"] in selected]
            records = order
            st.info("十种任务顺序的波动，非十个训练种子。仅导入明确标注的 SD，没有十次精确观测。")
            if order:
                a, b = st.columns(2)
                with a: show("顺序-A-Dice-SD", charts.order_sd(order, "A-Dice"))
                with b: show("顺序-BWTR-SD", charts.order_sd(order, "BWTR"))
            else:
                st.info("当前选择没有已报告的顺序研究方法。")
            with st.expander("任务顺序研究 · 查看原图"):
                picture("task_robustness_adice", "fig:benchmark-order-adice · PDF 第 1 页；Regu-LWF 对应 Regu-LwF。", pictures)
                picture("task_robustness_bwt_from_bottom", "fig:benchmark-order-bwtr · PDF 第 1 页；仅 SD 标注可精确导入。", pictures)
            st.markdown("**回放容量研究 · 原论文图**")
            picture("domain_memory_size_adice", "fig:benchmark-memory · PDF 第 1 页 · 当前附件没有可信逐点数值序列；不从曲线估计实验值。", pictures)
        elif view == "SAM扩展":
            records = sam_cells(registry)
            scope["filter_scope"] = "SAM 独立补充研究；主表的方法、类别、参照、SD 筛选不作用于这两种适配方式。"
            st.caption(scope["filter_scope"])
            a, b = st.columns(2)
            with a: show("SAM", charts.sam_heatmap(registry, "SAM"))
            with b: show("SAM-LoRA", charts.sam_heatmap(registry, "SAM-LoRA"))
            st.caption("行：完成训练阶段；列：测试任务。黑框对角线为当前任务，严格下三角为历史任务，严格上三角为未来域直接测试。两图共享 0–1 色标。")
            a, b = st.columns(2)
            with a: show("LoRA−SAM", charts.sam_heatmap(registry, "LoRA−SAM"))
            with b:
                summary = sam_summary(registry)
                st.markdown("**由论文已报告矩阵计算**")
                st.write(f"整体矩阵均值：{summary['SAM']['matrix_mean']:.4f} → {summary['SAM-LoRA']['matrix_mean']:.4f}。论文的 0.747 → 0.823 指此统计量。")
                st.write(f"最终 A-Dice：{summary['SAM']['final_A_Dice']:.4f} → {summary['SAM-LoRA']['final_A_Dice']:.4f}。")
                st.write("36 个阶段—任务单元：32 个增加、4 个下降。差值热图保留负增益；不是病例胜率。")
                st.caption("输入精度为原 PDF 标注的 3 位小数；派生值不代表更高实验精度。缺少随机初始化和独立训练参照，不计算 SAM E-FWT 或 RMA。")
            task = st.selectbox("观察测试任务", [f"T{i}" for i in range(1, 7)], key="ch03-sam-task")
            scope["test_task"] = task
            show("SAM阶段观测", charts.sam_curve(registry, task))
            st.caption(f"竖虚线为 {task} 首次学习阶段；此前是未来域直接测试。仅连接六个已报告观测，无平滑、补点或误差线。")
            with st.expander("查看 SAM 原论文图"):
                picture("SAM_confusion_matrix", "fig:benchmark-sam-lora · PDF 第 1 页 · 72 个明确标注值。", pictures)
        elif view == "数据与来源":
            st.caption("完整原文数值及独立 mean / sd 字段；支持列排序和表格内部横向滚动。当前筛选适用于主表。")
            table = []
            for row in sorted(rows, key=lambda r: -r["metrics"]["A-Dice"]["mean"]):
                line = {"方法": row["method_id"], "类别": FAMILIES[row["family"]]}
                for name, cell in row["metrics"].items():
                    line[name] = ("未报告 / 不适用" if cell["mean"] is None else
                        f"{cell['mean']:.3f} ± {cell['sd']:.3f}" if cell["sd"] is not None else f"{cell['mean']:g}")
                    if name == "DRR" and row["drr_star"]:
                        line[name] += " *"
                table.append(line)
            st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
            with st.expander("独立 mean / sd 与完整来源字段"):
                first = ["method_id", "metric", "mean", "sd", "status", "missing_reason", "source_raw", "unit", "scenario", "family"]
                frame = pd.DataFrame(records)
                st.dataframe(frame, column_order=first + [c for c in frame.columns if c not in first], hide_index=True, width="stretch")
            for metric, explanation in METRICS.items():
                st.markdown(f"**{metric}**　{explanation}")
            st.info("Organ-CL 未报告 Repl-DER++，不是得分为零。原表 -- 表示不适用或未报告；原文未细分时保持该状态。MPE/DRR 的 SD 为 null；0±0 为真实报告。")
            st.json({"schema": registry["schema"], "source_type": registry["source_type"], "platform_recomputed": False,
                "source_zip_sha256": registry["source_zip_sha256"], "source_hashes": registry["source_hashes"]}, expanded=False)
            with st.expander("逐行原文映射 · 46 条完整主表记录"):
                st.dataframe(pd.DataFrame(registry["source_table_mapping"]), hide_index=True, width="stretch")
            # A report from this view still contains a real chart, alongside the full data.
            if rows:
                show("数据摘要-A-Dice", charts.ranking(rows, sd=sd))

        with st.expander("协议与指标来源"):
            for key, explanation in PROTOCOL.items():
                if key not in SCENARIOS or key == scenario:
                    st.markdown(f"**{key}**　{explanation}")
            st.caption("主表来源：chapters/ch03_medcl_benchmark.tex · tab:benchmark-domain-results / class-results / organ-results。各标记的 hover 保留原文定位；数据与来源页提供完整 SHA-256。")
        with st.expander("导出当前视图 · CSV / JSON / 离线 HTML / SVG / PNG"):
            payload = dict(schema="medcl.ch03.export.v1", notice=NOTICE, source_type="thesis_reference", platform_recomputed=False,
                scope=scope, records=records, source_hashes=registry["source_hashes"], metrics=METRICS, protocol=PROTOCOL,
                memory_capacity_study=registry["memory_capacity_study"],
                sam_derived={"values": sam_summary(registry), "source_type": "derived_from_reported_matrix",
                    "source_precision": 3, "source_file": registry["sam_matrices"]["source_file"],
                    "note": "由论文已报告矩阵计算；显示小数位不代表更高实验精度。"} if view == "SAM扩展" else None)
            signature = json.dumps(scope, sort_keys=True)
            st.caption("离线 HTML 嵌入真实图表图片及当前数值、筛选范围、来源和指标解释。图表导出使用本地渲染器。")
            if st.button("生成当前视图导出", key="ch03-export"):
                try:
                    with st.spinner("正在本地生成图表与报告…"):
                        st.session_state["ch03-export-result"] = (signature, export_view(payload, figures, pictures))
                except Exception as exc:
                    st.error(f"导出失败：{exc}")
            saved = st.session_state.get("ch03-export-result")
            if saved and saved[0] == signature:
                result = saved[1]
                for ext, mime in [("csv", "text/csv"), ("json", "application/json"), ("html", "text/html")]:
                    st.download_button(f"下载 {ext.upper()}", result[ext], f"ch03-{scenario}-{view}.{ext}", mime=mime, key=f"ch03-download-{ext}")
                for name, rendered in result["charts"].items():
                    a, b = st.columns(2)
                    a.download_button(f"{name} · SVG", rendered["svg"], f"{name}.svg", "image/svg+xml", key=f"ch03-svg-{name}")
                    b.download_button(f"{name} · PNG", rendered["png"], f"{name}.png", "image/png", key=f"ch03-png-{name}")
