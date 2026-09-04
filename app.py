"""The local MedCL browser application. Long-running scoring belongs to the worker."""

from html import escape
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from medcl.benchmarks import INCREMENTS, KINDS, allowed_output_heads, catalog, config_path, readiness
from medcl_cornerstone import envelope_from_preview, render as render_volume
from medcl.examples import baseline_predictions, example_weights, pack_predictions, sample_manifest
from medcl.reports import compatible_result, compatibility, report_csv, report_html, report_json
from medcl.runner import load_preview
from medcl.sandbox import sandbox_available
from medcl.storage import get_job, initialize, job_dir, list_jobs, worker_alive
from medcl.submissions import ARCHITECTURES, submit

st.set_page_config(page_title="MedCL · 医学影像持续学习评测", page_icon="🔬", layout="wide")
st.html('''<style>
:root {
  --page-bg:#F6F8FB;--surface:#FFFFFF;--surface-subtle:#F9FAFC;
  --text:#17212B;--text-secondary:#667085;--text-tertiary:#8A94A3;
  --border:#E3E8EF;--border-strong:#D4DBE5;
  --primary:#2563EB;--primary-hover:#1D4ED8;--primary-soft:#EEF4FF;
  --success:#15803D;--success-soft:#ECFDF3;
  --warning:#A16207;--warning-soft:#FFFBEB;
  --danger:#B42318;--danger-soft:#FEF3F2;--viewer-bg:#070B12;
}
html,body,.stApp,input,button,textarea {font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;color:var(--text);}
.stApp,[data-testid="stAppViewContainer"] {background:var(--page-bg);}
.stMainBlockContainer {max-width:1760px;padding:1.25rem 2.5rem 4rem;}
[data-testid="stHeader"] {background:transparent;}
[data-testid="stToolbar"],[data-testid="stDecoration"],footer {display:none!important;}
h1 {font-size:2.35rem!important;font-weight:720!important;letter-spacing:-.045em!important;line-height:1.12!important;margin:.35rem 0 .4rem!important;color:var(--text)!important;}
h2 {font-size:1.45rem!important;font-weight:680!important;letter-spacing:-.025em!important;color:var(--text)!important;}
h3 {font-size:1.12rem!important;font-weight:650!important;color:var(--text)!important;}
h4 {font-size:.9rem!important;font-weight:680!important;color:#344054!important;letter-spacing:.01em!important;margin-top:.3rem!important;}
p,li {line-height:1.65;} label p {color:#344054!important;font-weight:520!important;}
[data-testid="stCaptionContainer"] p {color:var(--text-secondary)!important;font-size:.875rem!important;line-height:1.55!important;}
.topbar {display:flex;align-items:center;min-height:64px;border-bottom:1px solid var(--border);padding:0 0 14px;margin-bottom:16px;}
.brand {display:flex;align-items:center;gap:12px;}
.brand-mark {border:1px solid #BFD0FF;background:var(--primary-soft);width:40px;height:40px;border-radius:11px;display:grid;place-items:center;font-size:19px;font-weight:750;color:var(--primary);}
.brand strong {font-size:22px;letter-spacing:-.035em;color:var(--text);}.brand small{display:block;color:var(--text-secondary);font-size:12px;margin-top:1px;}
[data-testid="stButtonGroup"] [role="radiogroup"] {background:var(--surface);border:1px solid var(--border);border-radius:11px;padding:4px;box-shadow:0 1px 2px rgba(16,24,40,.03);}
[data-testid="stButtonGroup"] button {border:0!important;border-radius:7px!important;min-height:36px!important;color:var(--text-secondary)!important;}
[data-testid="stButtonGroup"] button[aria-checked="true"] {background:var(--text)!important;color:#fff!important;box-shadow:0 1px 2px rgba(16,24,40,.18)!important;}
[data-testid="stButtonGroup"] button[aria-checked="true"] p {color:#fff!important;}
[data-testid="stVerticalBlockBorderWrapper"] {background:var(--surface);border:1px solid var(--border)!important;border-radius:14px!important;box-shadow:0 1px 2px rgba(16,24,40,.025);transition:border-color .15s ease,box-shadow .15s ease;}
[data-testid="stVerticalBlockBorderWrapper"]:hover {border-color:var(--border-strong)!important;box-shadow:0 8px 24px rgba(16,24,40,.055);}
.task-heading {display:flex;align-items:center;gap:12px;margin:2px 0 8px;}.task-heading h3{font-size:1.16rem;margin:0;}.task-heading small{display:block;color:var(--text-secondary);font-size:12px;margin-top:2px;}
.task-symbol {width:38px;height:38px;border-radius:10px;background:var(--primary-soft);color:var(--primary);display:grid;place-items:center;font-weight:750;font-size:15px;flex:0 0 auto;}
.inline-note {border-left:3px solid var(--primary);background:var(--primary-soft);color:#344054;border-radius:0 8px 8px 0;padding:9px 12px;margin:8px 0 14px;font-size:13px;line-height:1.55;}
[data-testid="stMetric"] {background:var(--surface);border:1px solid var(--border);border-radius:13px;padding:16px 18px;min-height:108px;box-shadow:0 1px 2px rgba(16,24,40,.025);}
[data-testid="stMetricLabel"] p {color:var(--text-secondary)!important;font-size:.82rem!important;font-weight:560!important;}
[data-testid="stMetricValue"] {font-size:1.9rem!important;font-weight:680!important;letter-spacing:-.035em!important;color:var(--text)!important;}
.badge {display:inline-flex;align-items:center;border:1px solid var(--border);background:var(--surface-subtle);color:#475467;padding:3px 9px;font-size:12px;border-radius:999px;margin-right:6px;}
.badge.ready {background:var(--success-soft);border-color:#C7EAD4;color:var(--success);}.badge.pending{background:var(--warning-soft);border-color:#F4E2B7;color:var(--warning);}
.timeline {display:flex;flex-wrap:wrap;gap:10px;margin:14px 0 24px;}.task-step {display:flex;align-items:center;gap:9px;border:1px solid var(--border);border-radius:9px;background:var(--surface);padding:9px 13px;font-size:14px;}.task-step b {color:var(--primary);font-size:12px;}.task-step span{color:#344054;}
.stButton button,.stDownloadButton button {border-radius:9px!important;min-height:40px!important;font-weight:600!important;border-color:var(--border-strong)!important;box-shadow:none!important;}
.stButton button[kind="primary"],.stDownloadButton button[kind="primary"] {background:var(--primary)!important;border-color:var(--primary)!important;color:#fff!important;}
.stButton button[kind="primary"]:hover,.stDownloadButton button[kind="primary"]:hover {background:var(--primary-hover)!important;border-color:var(--primary-hover)!important;}
.stButton button:focus-visible,.stDownloadButton button:focus-visible {outline:3px solid #BFDBFE!important;outline-offset:2px!important;}
[data-testid="stAlert"] {background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:10px!important;padding:.72rem .9rem!important;color:#344054!important;box-shadow:none!important;}
[data-testid="stAlertContainer"] {background:transparent!important;padding:0!important;}
[data-testid="stFileUploaderDropzone"],input,textarea,[data-baseweb="select"]>div {background:var(--surface)!important;border-color:var(--border-strong)!important;border-radius:10px!important;}
[data-testid="stDataFrame"],[data-testid="stVegaLiteChart"] {background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:8px;overflow:hidden;}
[data-baseweb="tab-list"] {gap:22px;border-bottom:1px solid var(--border);}
[data-baseweb="tab"] {padding:12px 2px!important;color:var(--text-secondary)!important;font-weight:600!important;}
[aria-selected="true"][data-baseweb="tab"] {color:var(--primary)!important;}
[data-testid="stExpander"] {background:var(--surface);border-color:var(--border)!important;border-radius:10px!important;}
hr {border-color:var(--border)!important;}
@media(max-width:900px){.stMainBlockContainer{padding:1rem 1.25rem 3rem}h1{font-size:1.9rem!important}.task-step{padding:7px 10px}}
</style>''')

initialize()
SHOW_DEMOS = os.environ.get("MEDCL_SHOW_DEMOS") == "1"
STATUS = {"queued": "排队中", "running": "评测中", "completed": "已完成", "failed": "失败"}
NAV = ["任务中心", "评测记录", "方法比较"]


def navigate(page):
    st.session_state.nav = page


@st.cache_data(ttl=30)
def load_catalog(config_mtime):
    return catalog()


cfg = config_path()
try:
    benchmarks = load_catalog(cfg.stat().st_mtime_ns if cfg.is_file() else None)
except Exception:
    st.error("管理员基准配置不可读；请检查配置后刷新。服务器路径不会显示在网页。")
    st.stop()
lookup = {b["id"]: b for b in benchmarks}

st.html('<div class="topbar"><div class="brand"><div class="brand-mark">M</div><div><strong>MedCL</strong><small>医学影像持续学习评测</small></div></div></div>')
if st.session_state.get("nav") not in NAV:
    st.session_state.nav = NAV[0]
page = st.segmented_control("主导航", NAV, key="nav", label_visibility="collapsed", width="stretch")


def title(name, detail):
    st.title(name)
    st.caption(detail)


def timeline(order, b):
    names = {t["id"]: t["name"] for t in b["tasks"]}
    st.html('<div class="timeline">' + "".join(f'<div class="task-step"><b>阶段 {i + 1}</b><span>{escape(t)} · {escape(names[t])}</span></div>' for i, t in enumerate(order)) + '</div>')


def benchmark_visible(benchmark):
    return SHOW_DEMOS or (not benchmark.get("synthetic") and readiness(benchmark)[0])


def visible_jobs():
    return [job for job in list_jobs() if SHOW_DEMOS or not job["config"]["benchmark"].get("synthetic")]


def score_text(value, unit="fraction"):
    return "—" if value is None else f"{value:.3f} mm" if unit == "mm" else f"{value:.4f}"


def heatmap(matrix, columns, rows, direction="higher", title_text=""):
    values = [{"任务": t, "阶段 / 客户端": row, "分数": matrix[i][j],
               "显示": "—" if matrix[i][j] is None else f"{matrix[i][j]:.3f}",
               "状态": "未评测 / 缺少阶段 / 无测试样本" if matrix[i][j] is None else "实测"}
              for i, row in enumerate(rows) for j, t in enumerate(columns)]
    df = pd.DataFrame(values)
    domain = [0, 1] if direction == "higher" else [0, max([v["分数"] for v in values if v["分数"] is not None] or [1]) or 1]
    base = alt.Chart(df).encode(x=alt.X("任务:N", sort=columns, axis=alt.Axis(labelAngle=0, title=None)),
                                y=alt.Y("阶段 / 客户端:N", sort=rows, axis=alt.Axis(title=None)),
                                tooltip=["任务", "阶段 / 客户端", "显示", "状态"])
    rect = base.mark_rect(stroke="#ffffff", strokeWidth=3, cornerRadius=4).encode(
        color=alt.condition("isValid(datum.分数)", alt.Color("分数:Q", scale=alt.Scale(domain=domain, range=["#EEF4FF", "#2563EB"] if direction == "higher" else ["#2563EB", "#EEF4FF"]), legend=None), alt.value("#F2F4F7")))
    text = base.mark_text(fontSize=14).encode(text="显示:N", color=alt.condition(
        f"isValid(datum.分数) && datum.分数 {'>' if direction == 'higher' else '<'} {sum(domain)/2}", alt.value("white"), alt.value("#314957")))
    chart = (rect + text).properties(height=max(170, len(rows) * 52), title=title_text).configure_view(stroke=None).configure_axis(
        domain=False, tickColor="#D4DBE5", labelColor="#475467", titleColor="#344054")
    st.altair_chart(chart, width="stretch")


def task_center():
    chosen = st.session_state.get("selected_benchmark")
    if chosen in lookup and benchmark_visible(lookup[chosen]):
        if st.button("← 返回具体任务"):
            st.session_state.pop("selected_benchmark", None)
            st.rerun()
        new_evaluation(chosen, st.session_state.get("selected_supervision"))
        return
    st.session_state.pop("selected_benchmark", None)
    title("任务中心", "选择医学影像任务和持续学习场景。")
    selected_kind = st.session_state.get("selected_kind")
    if selected_kind not in KINDS:
        st.subheader("选择任务类型")
        cards = {
            "segmentation": ("分", "医学影像分割", "多场景持续学习", "域增量、类增量与任务增量；支持全监督和弱监督结果展示。"),
            "classification": ("类", "医学影像分类", "类别增量评测", "展示类别名称、整体准确率与逻辑客户端聚合结果。"),
            "registration": ("配", "医学影像配准", "任务增量评测", "对比固定、移动与配准后影像，展示标志点 TRE。"),
        }
        columns = st.columns(3, gap="medium")
        for column, (kind, (symbol, name, label, detail)) in zip(columns, cards.items()):
            with column:
                with st.container(border=True):
                    st.html(f'<div class="task-heading"><span class="task-symbol">{symbol}</span><div><h3>{name}</h3><small>{label}</small></div></div>')
                    st.write(detail)
                    if st.button(f"进入{KINDS[kind]}任务", key=f"enter-{kind}", type="primary" if kind == "segmentation" else "secondary", width="stretch"):
                        st.session_state.selected_kind = kind
                        st.rerun()
        completed = sum(job["status"] == "completed" for job in visible_jobs())
        if completed:
            st.caption(f"已完成评测 {completed} 条。")
        return

    if st.button("← 返回三类大任务"):
        st.session_state.pop("selected_kind", None)
        st.rerun()
    scenario_keys = {"segmentation": ["domain", "class", "task"], "classification": ["class"], "registration": ["task"]}[selected_kind]
    scenario_labels = [INCREMENTS[key] for key in scenario_keys]
    supervision = None
    scenario_column, supervision_column = st.columns(2, gap="large")
    with scenario_column:
        st.subheader(f"{KINDS[selected_kind]}持续学习场景")
        scenario_label = st.segmented_control("增量场景", scenario_labels, default=scenario_labels[0], key=f"scenario-{selected_kind}", width="stretch")
        scenario = scenario_keys[scenario_labels.index(scenario_label)]
        if selected_kind == "classification":
            st.caption("支持类别增量和逻辑客户端聚合评分，不包含联邦训练或通信。")
    with supervision_column:
        if selected_kind == "segmentation":
            st.subheader("训练监督方式")
            supervision_label = st.segmented_control("分割监督方式", ["全监督", "弱监督"], default="全监督", key="segmentation-supervision", width="stretch")
            supervision = {"全监督": "full", "弱监督": "weak"}[supervision_label]
            st.caption("监督方式由提交者声明；两者使用同一冻结测试集计算 Dice。")

    st.subheader("选择评测协议")
    shown = [b for b in benchmarks if b["kind"] == selected_kind and b["incremental"] == scenario]
    if SHOW_DEMOS:
        scope = st.segmented_control("数据范围", ["全部协议", "真实 / 待接入", "合成模拟"], default="全部协议", key=f"scope-{selected_kind}", width="stretch")
        if scope == "真实 / 待接入":
            shown = [b for b in shown if not b.get("synthetic")]
        elif scope == "合成模拟":
            shown = [b for b in shown if b.get("synthetic")]
    else:
        shown = [b for b in shown if benchmark_visible(b)]
    shown.sort(key=lambda b: (2 if b.get("synthetic") else 0 if readiness(b)[0] else 1, b["title"]))
    if not shown:
        st.info("当前没有可用的评测协议。")
    for b in shown:
        ok, reason = readiness(b)
        tasks = " → ".join(f"{t['id']} {t['name']}" for t in b["tasks"]) or "任务和测试资产待登记"
        with st.container(border=True):
            badges = f'<span class="badge">{INCREMENTS[b["incremental"]]}</span><span class="badge">{escape(b["metric"])}</span>'
            if SHOW_DEMOS:
                badges += f'<span class="badge {"ready" if ok else "pending"}">{"✓ " if ok else "○ "}{escape(reason)}</span><span class="badge">{"合成模拟" if b.get("synthetic") else "真实协议"}</span>'
            st.html(badges)
            st.markdown(f"### {b['title']}")
            st.write(b["description"])
            st.caption(tasks)
            if b.get("class_names"):
                st.caption("类别名称：" + "；".join(f"{k} {v}" for k, v in b["class_names"].items()))
            if st.button("配置此任务", key=f"configure-{b['id']}", disabled=not ok, type="primary" if ok else "secondary"):
                st.session_state.selected_benchmark = b["id"]
                st.session_state.selected_supervision = supervision
                st.rerun()
    if SHOW_DEMOS:
        with st.expander("开发模式说明"):
            st.write("待接入协议缺少真实测试资产；合成协议只用于验收工程链路。")


def new_evaluation(benchmark_id, training_supervision=None):
    title("配置与提交", "选择评测条件并上传模型或预测；提交后配置固定。")
    b = lookup[benchmark_id]
    ok, reason = readiness(b)
    st.caption(f"{KINDS[b['kind']]} · {INCREMENTS[b['incremental']]} · {b['description']}")
    if not ok:
        st.warning(reason + "。可先选择已就绪的真实基准或合成验收协议。")
        return
    if b.get("synthetic"):
        st.warning("当前是合成工程验收样例，所有结果均不属于真实科研结果。")
    if b["kind"] == "segmentation":
        training_supervision = training_supervision if training_supervision in ("full", "weak") else "full"
        supervision_name = "全监督" if training_supervision == "full" else "弱监督"
        st.html(f'<div class="inline-note">训练条件：<strong>{supervision_name}</strong>（提交者声明）。评分统一使用冻结完整测试标注。</div>')
    names = {t["id"]: t["name"] for t in b["tasks"]}
    standard = list(names)
    st.subheader("评测协议")
    custom = st.toggle("使用自定义任务顺序", key=f"custom-{benchmark_id}")
    order = standard
    if custom:
        order = [st.selectbox(f"阶段位置 {i + 1}", standard, index=i, format_func=lambda t: f"{t} · {names[t]}", key=f"order-{benchmark_id}-{i}")
                 for i in range(len(standard))]
        if len(set(order)) != len(order):
            st.error("任务 ID 重复；每个任务必须出现一次。")
    timeline(order, b)
    settings_column, submission_column = st.columns(2, gap="large")
    with settings_column:
        st.markdown("#### 评测设置")
        scenario = st.radio("评测场景", ["集中式", "逻辑客户端评测"], horizontal=True)
        clients = st.selectbox("逻辑客户端数", [2, 3, 4], index=1) if scenario != "集中式" else 1
        if clients > 1:
            st.caption(f"匿名病例固定划分至 {clients} 个逻辑客户端，仅汇总评分，不训练或聚合权重。")
        heads = allowed_output_heads(b)
        head = st.selectbox("输出头与任务信息", heads, format_func=lambda x: "共享输出头 / 全局类别编码" if x == "shared" else "任务指定输出头 / 已知任务 ID")
        unseen_key = f"evaluate-unseen-{benchmark_id}"
        unseen_enabled = b["allow_unseen"] and head == "shared"
        if unseen_key not in st.session_state or not unseen_enabled:
            st.session_state[unseen_key] = False
        unseen = st.checkbox("同时评测未见任务", disabled=not unseen_enabled, key=unseen_key)
        if not b["allow_unseen"]:
            st.caption("当前协议不评分未见任务。")
    with submission_column:
        st.markdown("#### 提交内容")
        method = st.text_input("方法 / 本次评测名称", value="", placeholder="例如：方法名称 · 最终模型 · 随机种子 42", max_chars=80)
        registration_volume = b["kind"] == "registration" and any(t.get("format") == "registration-volume" for t in b["tasks"])
        mode_options = ["预测文件"] if registration_volume else ["预测文件", "已支持结构的模型权重"]
        mode_label = st.radio("提交类型", mode_options, horizontal=True)
        mode = "predictions" if mode_label == "预测文件" else "model"
        if b["synthetic"]:
            provenance = "synthetic"
            st.caption("结果来源固定为合成工程验收。")
        else:
            provenance_labels = {
                "external_predictions_unknown": "外部预测 / 来源未知",
                "untrained_baseline": "未训练工程基线",
                "trained_model_declared": "已训练模型（提交者声明）",
            }
            provenance = st.selectbox("结果来源声明", list(provenance_labels), format_func=provenance_labels.get,
                                      key=f"provenance-{benchmark_id}")
            st.caption("平台验证测试评分，不验证训练过程。")
        architecture = None
        model_ok = True
        if mode == "model":
            model_ok = sandbox_available()
            architecture = st.selectbox("已审核模型结构", list(ARCHITECTURES[b["kind"]]), format_func=ARCHITECTURES[b["kind"]].get)
            st.caption("仅支持只读 float32 safetensors 与上列结构；其他模型请先生成预测。")
            if head != "shared":
                st.warning("已审核模型仅支持共享输出头；当前条件请上传预测文件。")
            if model_ok:
                st.caption("模型隔离检查已通过。")
            else:
                st.warning("当前环境不支持隔离模型运行，请上传预测文件。")
        else:
            st.caption("支持 JSON 或 NPZ/ZIP 预测文件。客户端划分由平台完成。")
            if registration_volume:
                st.caption("可附带对齐固定网格的配准后影像与变形标签，仅用于可视化。")
    st.subheader("上传阶段结果")
    scope = st.radio("可提供的阶段", ["仅最终阶段", "多个 / 部分阶段"], horizontal=True)
    stages = [len(order)] if scope == "仅最终阶段" else st.multiselect("已有阶段位置", list(range(1, len(order) + 1)), default=[len(order)])
    st.caption("评分将输出任务主指标、最终平均、后向迁移、遗忘、前向迁移和相对后向迁移；条件不足时显示为不可计算。")
    files = []
    for stage in sorted(stages):
        seen_names = " → ".join(order[:stage])
        up = st.file_uploader(f"阶段 {stage} · 已见任务 {seen_names}", type=["safetensors"] if mode == "model" else ["json", "npz", "zip"],
                              accept_multiple_files=False, key=f"upload-{benchmark_id}-{mode}-{stage}", disabled=not model_ok)
        if up is not None:
            files.append({"stage": stage, "name": up.name, "data": up.getvalue()})
    with st.expander("样本索引、文件格式与可下载示例"):
        st.write("使用平台生成的匿名样本 ID；预测必须覆盖该阶段协议要求的全部任务与样本。隐藏标签不包含在索引中。")
        if st.button("准备样本索引", key=f"manifest-{benchmark_id}"):
            try:
                st.session_state[f"manifest-data-{benchmark_id}"] = sample_manifest(b)
            except Exception:
                st.error("测试资产索引不可读，请管理员检查数据挂载。")
        manifest_data = st.session_state.get(f"manifest-data-{benchmark_id}")
        if manifest_data:
            st.download_button("下载匿名样本索引 JSON", manifest_data, f"{benchmark_id}-sample-index.json", "application/json")
        st.code('{"schema":"medcl.predictions.v1","tasks":{"T1":{"sample_ids":["T1-s000000"],"predictions":[0]}}}', language="json")
        if b.get("synthetic"):
            example = pack_predictions(baseline_predictions(b, standard), as_json=b["kind"] != "segmentation")
            suffix = "npz" if b["kind"] == "segmentation" else "json"
            st.download_button("下载最终阶段合成预测示例", example, f"{benchmark_id}-final.{suffix}", "application/octet-stream")
            st.download_button("下载未训练的结构验收权重", example_weights(b["kind"]), f"{benchmark_id}-untrained.safetensors", "application/octet-stream")
            st.caption("示例权重仅用于工程验收，从未训练；不能作为方法结果。")
    st.subheader("开始评测")
    if clients > 1 and mode == "predictions":
        st.caption("预测模式由提交者声明同一阶段各客户端预测来自同一全局模型；平台不要求训练日志，也不伪称已验证模型来源。")
    if not worker_alive():
        st.warning("评分服务暂不可用，请联系管理员启动。")
    ready = bool(method.strip()) and bool(stages) and len(files) == len(stages) and len(set(order)) == len(order) and model_ok and (mode != "model" or head == "shared") and (not unseen or unseen_enabled) and worker_alive()
    if st.button("提交并开始评测", type="primary", disabled=not ready):
        try:
            job_id = submit(b, method=method, order=order, uploads=files, mode=mode, architecture=architecture,
                            clients=clients, evaluate_unseen=unseen, output_head=head,
                            training_supervision=training_supervision, provenance=provenance)
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            st.error("提交保存失败；未开始评测，请联系管理员检查私有存储。")
        else:
            st.session_state.selected_job = job_id
            st.session_state.submitted_notice = job_id
            st.success("已提交，评分将在后台完成。")
            st.button("查看此次评测", on_click=navigate, args=("评测记录",), type="primary")


def preview_arrays(job_id, reference):
    return load_preview(job_dir(job_id), reference)


def segmentation_fallback(image, prediction):
    """Build one prediction-only static slice when WebGL is unavailable."""
    counts = np.count_nonzero(prediction, axis=(1, 2))
    index = int(np.argmax(counts)) if np.any(counts) else prediction.shape[0] // 2
    original = np.asarray(image[index], dtype=np.uint8)
    overlay = np.repeat(original[..., None], 3, axis=2)
    foreground = prediction[index] > 0
    overlay[foreground] = (0.35 * overlay[foreground] + 0.65 * np.array([38, 200, 122])).astype(np.uint8)
    return original, overlay, index


def result_view(job):
    result = compatible_result(job["result"])
    config = result["config"]
    b = config["benchmark"]
    if b.get("synthetic"):
        st.warning("合成工程验收结果 · 不得作为医学或论文实验结果")
    provenance = ({"category": "synthetic", "statement": "由已校验的合成协议决定；仅用于工程验收"}
                  if b.get("synthetic") else config.get("provenance", {
                      "category": "external_predictions_unknown", "statement": "旧记录未保存结构化来源；按未知外部来源显示"}))
    if not isinstance(provenance, dict):
        provenance = {"category": "external_predictions_unknown", "statement": "结果来源格式无效；按未知外部来源显示"}
    if provenance.get("category") != "synthetic":
        st.warning(provenance.get("statement", "结果来源未经平台验证"))
    st.subheader(config["method"])
    supervision = {"full": "全监督", "weak": "弱监督", "not-declared": "未声明", "not-applicable": "不适用"}.get(config.get("training_supervision"), "未声明")
    detail = f"{b['title']}" + (f" · {supervision}分割" if b["kind"] == "segmentation" else "")
    if SHOW_DEMOS:
        detail += f" · {b['version']} · {config['mode']} · 配置已冻结"
    st.caption(detail)
    timeline(config["order"], b)
    if SHOW_DEMOS:
        st.caption(config["conditions"])
    summary = result["continual"]["global"]
    columns = st.columns(5)
    labels = {"Final average": "最终任务宏平均", "BWT": "后向迁移 BWT", "Forgetting": "遗忘", "FWT": "前向迁移 FWT", "BWTR": "相对后向迁移 BWTR"}
    for col, metric in zip(columns, labels):
        col.metric(labels[metric], score_text(summary[metric]["value"], b["unit"]))
        if summary[metric]["value"] is None:
            col.caption("条件不足")
    st.caption("— 表示当前阶段或参照条件不足；详细原因见阶段评测。")
    detail_label = "类别名称" if b["kind"] == "classification" else "病例结果与可视化"
    tab_matrix, tab_clients, tab_cases, tab_protocol = st.tabs(["阶段评测", "客户端对比", detail_label, "评测信息"])
    with tab_matrix:
        client_id = st.selectbox("查看结果层级", list(result["matrices"]), format_func=lambda x: "全体测试样本" if x == "global" else f"逻辑客户端 {x}")
        heatmap(result["matrices"][client_id], config["order"], [f"阶段 {s}" for s in range(1, len(config["order"]) + 1)], b["direction"], b["metric"] + (" ↓" if b["direction"] == "lower" else " ↑"))
        st.caption("— 表示未提交、不允许评测或无样本，不代表 0。")
        detail = pd.DataFrame([{"指标": k, "值": score_text(v["value"], b["unit"]), "条件 / 原因": v["reason"]} for k, v in result["continual"][client_id].items()])
        st.dataframe(detail, hide_index=True, width="stretch")
    with tab_clients:
        if config["clients"] == 1:
            st.info("本次为集中式评测。新建评测时可选择固定逻辑客户端场景。")
        if SHOW_DEMOS:
            st.caption(config["client_split"]["source"])
        stage = st.select_slider("查看阶段", options=list(range(1, len(config["order"]) + 1)), value=max(config["stages"]))
        client_ids = [x for x in result["matrices"] if x != "global"]
        heatmap([result["matrices"][c][stage - 1] for c in client_ids], config["order"], client_ids, b["direction"], "客户端 × 测试任务")
        stats = [r for r in result["federated"] if r["stage"] == stage]
        if stats:
            frame = pd.DataFrame(stats).rename(columns={"task_id": "任务", "client_macro": "客户端宏平均", "sample_weighted_accuracy": "按样本数加权准确率", "worst_client": "最低性能客户端", "client_std": "客户端标准差", "client_gap": "客户端极差", "available_clients": "有样本客户端", "total_clients": "客户端总数", "stage": "阶段"})
            st.dataframe(frame, hide_index=True, width="stretch")
            st.caption("分类按客户端样本数加权；TRE 越低越好。")
            dist = pd.DataFrame([r for r in result.get("distributions", []) if r.get("stage") == stage])
            if dist.empty:
                st.info("此阶段没有可展示的客户端样本量。")
            else:
                chart_data = dist.rename(columns={"client_id": "逻辑客户端", "n_samples": "样本数", "task_id": "任务"})
                chart = alt.Chart(chart_data).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(x="逻辑客户端:N", y="样本数:Q",
                    color=alt.Color("任务:N", scale=alt.Scale(range=["#2563EB", "#60A5FA", "#0F766E", "#7C3AED", "#475467", "#93C5FD"])),
                    tooltip=["逻辑客户端", "任务", "样本数"]).properties(height=250)
                st.altair_chart(chart, width="stretch")
                st.dataframe(dist.rename(columns={"client_id": "逻辑客户端", "n_samples": "样本数", "task_id": "任务", "stage": "阶段"}), hide_index=True, width="stretch")
        else:
            st.info("此阶段未提交，没有可展示的客户端统计。")
    with tab_cases:
        stage_control, task_control = st.columns(2, gap="medium")
        with stage_control:
            stage = st.selectbox("可视化阶段", sorted(config["stages"]), index=len(config["stages"]) - 1, key=f"case-stage-{job['id']}")
        cases_result = result.get("cases", []) if b["kind"] != "classification" else []
        available_tasks = [tid for tid in config["order"] if any(c["task_id"] == tid and c["stage"] == stage for c in cases_result)]
        with task_control:
            task_id = st.selectbox("类别任务" if b["kind"] == "classification" else "病例任务", available_tasks or config["order"], key=f"case-task-{job['id']}")
        cases = [c for c in cases_result if c["task_id"] == task_id and c["stage"] == stage]
        task = next(t for t in b["tasks"] if t["id"] == task_id)
        if b["kind"] == "classification":
            registered = lookup.get(b["id"], {})
            class_names = b.get("class_names") or (registered.get("class_names", {}) if registered.get("version") == b.get("version") else {})
            st.subheader("类别名称")
            st.dataframe(pd.DataFrame([{"类别编号": value, "类别名称": class_names.get(str(value), f"类别 {value}")} for value in task.get("classes", [])]), hide_index=True, width="stretch")
            st.caption("分类只发布任务与逻辑客户端聚合准确率；不显示或下载逐样本正误、样本 ID、病例 ID 或隐藏标签频数。")
        previews = [v for v in result.get("visualizations", []) if v["task_id"] == task_id and v["stage"] == stage]
        if b["kind"] in ("segmentation", "registration"):
            st.subheader("预测结果可视化")
            if previews:
                preview = previews[st.selectbox("选择预览病例", range(len(previews)), format_func=lambda i: f"{previews[i]['case_id']} · {b['metric']} {score_text(previews[i]['score'], b['unit'])}", key=f"preview-{job['id']}-{stage}-{task_id}")]
                try:
                    arrays = preview_arrays(job["id"], preview)
                    if b["kind"] == "segmentation":
                        if preview["kind"] == "segmentation-volume":
                            try:
                                render_volume(envelope_from_preview(preview, arrays), key=f"volume-{job['id']}-{stage}-{task_id}-{preview['case_id']}")
                            except (OSError, RuntimeError, TypeError, ValueError):
                                st.warning("三维组件未能挂载；下方仍保留不含真值的静态切片。")
                            original, overlay, slice_index = segmentation_fallback(arrays["image_volume"], arrays["prediction_volume"])
                        else:
                            original, overlay, slice_index = arrays["original"], arrays["overlay"], None
                        left, right = st.columns(2)
                        suffix = f" · Z={slice_index}" if slice_index is not None else ""
                        left.image(original, caption="原始测试切片" + suffix, width="stretch")
                        right.image(overlay, caption="预测遮罩叠加（绿色）" + suffix, width="stretch")
                        st.metric("该病例前景 Dice", score_text(preview["score"]))
                        st.caption("浏览器只接收原始影像与预测，不接收隐藏测试真值。")
                    else:
                        if preview["kind"] == "registration-volume":
                            try:
                                render_volume(envelope_from_preview(preview, arrays), key=f"volume-{job['id']}-{stage}-{task_id}-{preview['case_id']}")
                            except (OSError, RuntimeError, TypeError, ValueError):
                                st.warning("三维组件未能挂载；下方仍保留静态中心切片。")
                            center = arrays["fixed_volume"].shape[0] // 2
                            names = [("固定影像", arrays["fixed_volume"]), ("移动影像", arrays["moving_volume"])]
                            if "registered_volume" in arrays:
                                names.append(("提交的配准后影像", arrays["registered_volume"]))
                            columns = st.columns(len(names))
                            for column, (name, volume) in zip(columns, names):
                                column.image(volume[center], caption=f"{name} · Z={center}", width="stretch")
                            moving, prediction = arrays["moving_points"], arrays["predicted_points"]
                            if "registered_volume" not in arrays:
                                st.info("未提交配准后影像；三维查看器默认对比固定影像和移动影像。")
                        else:
                            moving, prediction = arrays["moving"], arrays["prediction"]
                        if moving.ndim != 2 or prediction.shape != moving.shape or moving.shape[1] < 2:
                            raise ValueError("配准预览维度无效")
                        points = [{"标志点": str(i), "位置": name, "步骤": step, "x": float(array[i, 0]), "y": float(array[i, 1])}
                                  for i in range(len(moving)) for step, (name, array) in enumerate((("移动点", moving), ("预测配准点", prediction)))]
                        frame = pd.DataFrame(points)
                        base = alt.Chart(frame).encode(x=alt.X("x:Q", scale=alt.Scale(zero=False)), y=alt.Y("y:Q", scale=alt.Scale(zero=False)))
                        chart = base.mark_line(color="#9aaab3").encode(detail="标志点:N", order="步骤:Q") + base.mark_point(size=80).encode(color=alt.Color("位置:N", scale=alt.Scale(range=["#708c9d", "#19a179"])), shape="位置:N", tooltip=["标志点", "位置", "x", "y"])
                        st.altair_chart(chart.properties(height=420), width="stretch")
                        st.metric("该病例 TRE", score_text(preview["score"], b["unit"]))
                        if preview["kind"] == "registration-volume":
                            st.caption("浏览器不接收隐藏固定点或真值分割；TRE 仅由预测标志点计算。")
                        else:
                            st.caption("该协议没有三维影像，仅显示移动点至预测配准点的 XY 投影；隐藏固定点只用于 TRE 评分。")
                except (OSError, ValueError, KeyError):
                    st.warning("私有可视化文件不可读；数值评分仍以已保存结果为准。")
            else:
                st.info("该记录没有可视化预览；新建的分割 / 配准评测每阶段、每任务最多生成 3 个私有病例预览。")
        if b["kind"] != "classification":
            st.subheader("病例数值")
            if cases:
                case_frame = pd.DataFrame(cases).drop(columns=["case_index"], errors="ignore").rename(columns={
                    "n_samples": "样本数", "score": b["metric"], "per_class": "分类别指标", "stage": "阶段",
                    "task_id": "任务", "case_id": "病例", "client_id": "逻辑客户端"})
                st.dataframe(case_frame, hide_index=True, width="stretch", height=340)
            else:
                st.info("该任务在选定阶段没有病例结果。")
    with tab_protocol:
        if SHOW_DEMOS:
            st.json(config)
        else:
            st.json({"评测协议": b["title"], "任务顺序": config["order"], "提交阶段": config["stages"],
                     "评测场景": "集中式" if config["clients"] == 1 else f"{config['clients']} 个逻辑客户端",
                     "输出头": config["output_head"], "监督方式": supervision})
        for warning in result["warnings"]:
            st.caption(warning)
    st.divider()
    st.subheader("下载评测报告")
    columns = st.columns(3)
    stem = f"medcl-{job['id'][:8]}"
    columns[0].download_button("下载 CSV 评分表", report_csv(result), stem + ".csv", "text/csv", width="stretch")
    columns[1].download_button("下载 JSON 完整报告", report_json(result), stem + ".json", "application/json", width="stretch")
    columns[2].download_button("下载 HTML 阅读版", report_html(result), stem + ".html", "text/html", width="stretch")


def records():
    title("评测记录", "查看评测进度、结果和报告。")
    jobs = visible_jobs()
    if not jobs:
        st.info("还没有评测记录。先提交一个最终模型或预测即可，无需训练日志。")
        st.button("新建第一条评测", on_click=navigate, args=("任务中心",), type="primary")
        return
    options = [j["id"] for j in jobs]
    mapping = {j["id"]: j for j in jobs}
    selected = st.session_state.get("selected_job", options[0])
    job_id = st.selectbox("选择评测记录", options, index=options.index(selected) if selected in options else 0,
                          format_func=lambda jid: f"{mapping[jid]['config']['method']} · {STATUS[mapping[jid]['status']]} · {jid[:8]}")
    st.session_state.selected_job = job_id
    active = mapping[job_id]["status"] in ("queued", "running")

    @st.fragment(run_every="5s" if active else None)
    def live_result():
        job = get_job(job_id)
        state = STATUS[job["status"]]
        st.caption(f"提交于 {job['created_at']} · 更新于 {job['updated_at']} (UTC)")
        if job["status"] == "failed":
            st.error(f"{state}：{job['message']}")
            if SHOW_DEMOS:
                st.json(job["config"], expanded=False)
        elif job["status"] in ("queued", "running"):
            st.info(f"{state}：{job['message']}")
            st.progress(job["progress"])
            st.caption("状态每 5 秒自动刷新；可以离开页面，评分会继续。")
        else:
            if active:
                st.rerun()
            st.success("评分完成 · 可查看结果并下载报告")
            result_view(job)
    live_result()
    st.button("刷新记录列表")


def compare():
    title("方法比较", "仅比较评测条件一致的结果；不同任务类型不合成总分。")
    jobs = [job for job in visible_jobs() if job["status"] == "completed"]
    if len(jobs) < 2:
        st.info("至少需要两条已完成且协议相容的评测。不会使用示意分数填充对比。")
        return
    mapping = {j["id"]: j for j in jobs}
    selected = st.multiselect("选择 2–4 条已完成评测", list(mapping), max_selections=4,
                             format_func=lambda jid: f"{mapping[jid]['config']['method']} · {mapping[jid]['config']['benchmark']['title']} · {jid[:8]}")
    if len(selected) < 2:
        return
    if len({compatibility(mapping[jid]["config"]) for jid in selected}) != 1:
        st.error("所选结果的评测条件不一致，无法直接比较。")
        return
    b = mapping[selected[0]]["config"]["benchmark"]
    if b.get("synthetic"):
        st.warning("合成验收对比，不代表真实方法性能。")
    rows = []
    provenance_labels = {
        "synthetic": "合成工程验收",
        "untrained_baseline": "未训练工程基线",
        "trained_model_declared": "已训练模型（提交者声明）",
        "external_predictions_unknown": "外部预测 / 来源未知",
    }
    for jid in selected:
        result = compatible_result(get_job(jid)["result"])
        provenance = result["config"]["provenance"]["category"]
        for tid, value in zip(result["config"]["order"], result["matrices"]["global"][-1]):
            rows.append({"方法": f"{result['config']['method']} · {jid[:6]}",
                         "来源": provenance_labels.get(provenance, "未知来源"), "任务": tid, b["metric"]: value})
    df = pd.DataFrame(rows)
    leaderboard = df.pivot(index=["方法", "来源"], columns="任务", values=b["metric"]).reset_index()
    leaderboard.columns.name = None
    st.dataframe(leaderboard, hide_index=True, width="stretch")
    chart = alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X(f"{b['metric']}:Q", title=b["metric"]), y=alt.Y("任务:N", title=None),
        color=alt.Color("方法:N", scale=alt.Scale(range=["#2563EB", "#60A5FA", "#0F766E", "#7C3AED"])),
        yOffset="方法:N", tooltip=list(df.columns)).properties(height=max(300, len(b["tasks"]) * 90)).configure_view(stroke=None).configure_axis(
            domain=False, gridColor="#E3E8EF", tickColor="#D4DBE5", labelColor="#475467", titleColor="#344054")
    st.altair_chart(chart, width="stretch")
    st.caption("比较最终阶段各任务；来源类别与方法名分开显示。缺少最终阶段的记录保留为空，不拿最后可见阶段代替最终阶段。")


{"任务中心": task_center, "评测记录": records, "方法比较": compare}[page]()
