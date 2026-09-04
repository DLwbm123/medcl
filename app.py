"""The local MedCL browser application. Long-running scoring belongs to the worker."""

from html import escape

import altair as alt
import pandas as pd
import streamlit as st

from medcl.benchmarks import INCREMENTS, KINDS, catalog, config_path, readiness
from medcl.examples import baseline_predictions, example_weights, pack_predictions, sample_manifest
from medcl.reports import compatible_result, compatibility, report_csv, report_html, report_json
from medcl.runner import load_preview
from medcl.sandbox import sandbox_available
from medcl.storage import get_job, initialize, job_dir, list_jobs, worker_alive
from medcl.submissions import ARCHITECTURES, submit

st.set_page_config(page_title="MedCL · 医学影像持续学习评测", page_icon="🔬", layout="wide")
st.html('''<style>
html,body,.stApp,input,button,textarea {font-family: -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;}
.stMainBlockContainer {max-width:1180px;padding-top:1.4rem;padding-bottom:3rem;}
h1 {font-size:2rem!important;font-weight:650!important;letter-spacing:-.035em;}
h2 {font-size:1.3rem!important;} h3 {font-size:1.1rem!important;}
[data-testid="stMetric"] {background:#fff;border:1px solid #dce4e8;border-radius:8px;padding:14px 18px;}
[data-testid="stMetricValue"] {font-size:1.75rem!important;}
.topbar {display:flex;align-items:center;justify-content:space-between;gap:24px;border-bottom:1px solid #dce4e8;padding:4px 0 15px;margin-bottom:12px;}
.brand {display:flex;align-items:center;gap:12px;margin:0;}
.brand-mark {border:1px solid #708c9d;background:#e0e9ed;width:40px;height:40px;border-radius:9px;display:grid;place-items:center;font-size:24px;color:#3b5e74;}
.brand strong {font-size:24px;letter-spacing:-.03em;}.brand small{display:block;color:#617382;font-size:12px;}
.runtime {text-align:right;color:#526b7a;font-size:13px;line-height:1.55}.runtime b{color:#355d4a}
.eyebrow {color:#627785;letter-spacing:.1em;font-size:12px;font-weight:650;margin-bottom:8px;}
.protocol-card {background:white;border:1px solid #dce4e8;border-radius:8px;padding:20px;margin:4px 0 10px;}
.protocol-card h3 {margin:10px 0 6px;}.protocol-card p{font-size:14px;line-height:1.7;color:#4c6372;margin:8px 0;}
.badge {display:inline-block;border:1px solid #cbd7dd;background:#edf2f5;color:#38586e;padding:2px 9px;font-size:12px;border-radius:4px;margin-right:6px;}
.badge.ready {background:#e8f0ed;border-color:#c8d9d1;color:#355d4a;}.badge.pending{background:#f7f1e7;border-color:#e1d4bb;color:#795c2e;}
.timeline {display:flex;flex-wrap:wrap;gap:9px;margin:12px 0 22px;}.task-step {display:flex;align-items:center;gap:9px;border:1px solid #cfdbe1;border-radius:6px;background:#fff;padding:9px 14px;font-size:14px;}.task-step b {color:#486b81;font-size:12px;}.task-step span{color:#243746;}
.subtle {color:#627785;font-size:14px;line-height:1.7;}
.stButton button,.stDownloadButton button {border-radius:6px;min-height:40px;}
@media(max-width:700px){.stMainBlockContainer{padding:1rem}.topbar{align-items:flex-start}.runtime{font-size:11px}.task-step{padding:7px 10px}h1{font-size:1.6rem!important}}
</style>''')

initialize()
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

alive = worker_alive()
st.html(f'<div class="topbar"><div class="brand"><div class="brand-mark">M</div><div><strong>MedCL</strong><small>医学影像持续学习评测</small></div></div><div class="runtime"><b>{"● 评分服务在线" if alive else "○ 评分服务离线"}</b><br>本地 evaluation · 不训练 / 不调参 / 不聚合权重</div></div>')
if st.session_state.get("nav") not in NAV:
    st.session_state.nav = NAV[0]
page = st.segmented_control("主导航", NAV, key="nav", label_visibility="collapsed", width="stretch")
st.caption("页面自上而下递进；单机多客户端是逻辑评分模拟，不代表多医院部署。")


def title(name, detail):
    st.html('<div class="eyebrow">MEDCL / EVALUATION WORKSPACE</div>')
    st.title(name)
    st.caption(detail)


def timeline(order, b):
    names = {t["id"]: t["name"] for t in b["tasks"]}
    st.html('<div class="timeline">' + "".join(f'<div class="task-step"><b>阶段 {i + 1:02d}</b><span>{escape(t)} · {escape(names[t])}</span></div>' for i, t in enumerate(order)) + '</div>')


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
    rect = base.mark_rect(stroke="#ffffff", strokeWidth=3, cornerRadius=3).encode(
        color=alt.condition("isValid(datum.分数)", alt.Color("分数:Q", scale=alt.Scale(domain=domain, range=["#e9f0f2", "#3d657e"] if direction == "higher" else ["#3d657e", "#e9f0f2"]), legend=None), alt.value("#eceff1")))
    text = base.mark_text(fontSize=14).encode(text="显示:N", color=alt.condition(
        f"isValid(datum.分数) && datum.分数 {'>' if direction == 'higher' else '<'} {sum(domain)/2}", alt.value("white"), alt.value("#314957")))
    chart = (rect + text).properties(height=max(160, len(rows) * 48), title=title_text).configure_view(stroke=None)
    st.altair_chart(chart, width="stretch")


def task_center():
    chosen = st.session_state.get("selected_benchmark")
    if chosen in lookup:
        if st.button("← 返回具体任务"):
            st.session_state.pop("selected_benchmark", None)
            st.rerun()
        new_evaluation(chosen, st.session_state.get("selected_supervision"))
        return
    st.session_state.pop("selected_benchmark", None)
    title("任务中心", "按医学影像大任务、持续学习场景和具体协议自上而下选择。")
    selected_kind = st.session_state.get("selected_kind")
    if selected_kind not in KINDS:
        st.subheader("一级 / 选择三类大任务")
        cards = {
            "segmentation": ("🧩", "医学影像分割", "优先完善：域增量、类增量、任务增量；全监督 / 弱监督；病例 Dice 与预测叠加图。"),
            "classification": ("🧬", "医学影像分类", "类别增量；显示类别语义名称；支持固定逻辑客户端的联邦评分模拟。"),
            "registration": ("🗺️", "医学影像配准", "任务增量；展示移动点与预测配准点；服务端计算 TRE。"),
        }
        for kind, (icon, name, detail) in cards.items():
            with st.container(border=True):
                st.markdown(f"### {icon} {name}")
                st.write(detail)
                if st.button(f"进入{KINDS[kind]}任务", key=f"enter-{kind}", type="primary" if kind == "segmentation" else "secondary"):
                    st.session_state.selected_kind = kind
                    st.rerun()
        st.caption(f"当前可用真实协议 {sum(readiness(b)[0] and not b.get('synthetic') for b in benchmarks)} 个；已完成评测 {sum(j['status'] == 'completed' for j in list_jobs())} 条。")
        return

    if st.button("← 返回三类大任务"):
        st.session_state.pop("selected_kind", None)
        st.rerun()
    st.subheader(f"二级 / {KINDS[selected_kind]}的持续学习场景")
    scenario_keys = {"segmentation": ["domain", "class", "task"], "classification": ["class"], "registration": ["task"]}[selected_kind]
    scenario_labels = [INCREMENTS[key] for key in scenario_keys]
    scenario_label = st.segmented_control("增量场景", scenario_labels, default=scenario_labels[0], key=f"scenario-{selected_kind}", width="stretch")
    scenario = scenario_keys[scenario_labels.index(scenario_label)]
    if selected_kind == "classification":
        st.info("分类当前开放类别增量；配置时可选固定逻辑客户端，展示客户端宏平均、样本加权准确率、最差客户端与差异；不伪称有联邦训练或通信。")

    supervision = None
    if selected_kind == "segmentation":
        st.subheader("三级 / 外部训练监督方式")
        supervision_label = st.segmented_control("分割监督方式", ["全监督", "弱监督"], default="全监督", key="segmentation-supervision", width="stretch")
        supervision = {"全监督": "full", "弱监督": "weak"}[supervision_label]
        st.caption("这是提交者声明的模型训练条件；两者都使用同一冻结、完整标注的测试集计算 Dice，平台不根据训练日志自行判定。")

    st.subheader(f"{'四' if selected_kind == 'segmentation' else '三'}级 / 选择具体任务")
    scope = st.segmented_control("数据范围", ["全部协议", "真实 / 待接入", "合成模拟"], default="全部协议", key=f"scope-{selected_kind}", width="stretch")
    shown = [b for b in benchmarks if b["kind"] == selected_kind and b["incremental"] == scenario]
    if scope == "真实 / 待接入":
        shown = [b for b in shown if not b.get("synthetic")]
    elif scope == "合成模拟":
        shown = [b for b in shown if b.get("synthetic")]
    shown.sort(key=lambda b: (2 if b.get("synthetic") else 0 if readiness(b)[0] else 1, b["title"]))
    if not shown:
        st.info("该场景尚无已登记协议。")
    for b in shown:
        ok, reason = readiness(b)
        tasks = " → ".join(f"{t['id']} {t['name']}" for t in b["tasks"]) or "任务和测试资产待登记"
        with st.container(border=True):
            st.html(f'<span class="badge">{INCREMENTS[b["incremental"]]}</span><span class="badge {"ready" if ok else "pending"}">{"✓ " if ok else "○ "}{escape(reason)}</span><span class="badge">{"合成模拟" if b.get("synthetic") else "真实协议"}</span>')
            st.markdown(f"### {b['title']}")
            st.write(b["description"])
            st.caption(f"{tasks} · {b['metric']} · {b['version']}")
            if b.get("class_names"):
                st.caption("类别名称：" + "；".join(f"{k} {v}" for k, v in b["class_names"].items()))
            if st.button("配置此任务", key=f"configure-{b['id']}", disabled=not ok, type="primary" if ok else "secondary"):
                st.session_state.selected_benchmark = b["id"]
                st.session_state.selected_supervision = supervision
                st.rerun()
    with st.expander("增量场景与数值边界"):
        st.write("域增量是输入中心 / 分布变化；类别增量是类别集合扩展；任务增量是器官、模态或分析目标变化。")
        st.write("没有真实测试资产的协议明确标记待接入；合成协议只验收工程链路，不补造科研结果。")


def new_evaluation(benchmark_id, training_supervision=None):
    title("配置与提交", "无需训练日志或用户编写的配置清单。提交后协议冻结，改变顺序需新建评测。")
    b = lookup[benchmark_id]
    ok, reason = readiness(b)
    st.caption(f"{b['version']} · {KINDS[b['kind']]} / {INCREMENTS[b['incremental']]} · {b['description']}")
    if not ok:
        st.warning(reason + "。可先选择已就绪的真实基准或合成验收协议。")
        return
    if b.get("synthetic"):
        st.warning("当前是合成工程验收样例，所有结果均不属于真实科研结果。")
    if b["kind"] == "segmentation":
        training_supervision = training_supervision if training_supervision in ("full", "weak") else "full"
        st.info(f"当前分割训练条件：{'**全监督**' if training_supervision == 'full' else '**弱监督**'}（提交者声明）。评分统一使用冻结完整测试标注。")
    names = {t["id"]: t["name"] for t in b["tasks"]}
    standard = list(names)
    st.subheader("01 / 固定评测协议")
    custom = st.toggle("使用自定义任务顺序", key=f"custom-{benchmark_id}")
    order = standard
    if custom:
        order = [st.selectbox(f"阶段位置 {i + 1}", standard, index=i, format_func=lambda t: f"{t} · {names[t]}", key=f"order-{benchmark_id}-{i}")
                 for i in range(len(standard))]
        if len(set(order)) != len(order):
            st.error("任务 ID 重复；每个任务必须出现一次。")
    timeline(order, b)
    scenario = st.radio("评测场景", ["集中式", "联邦持续评测（逻辑客户端）"], horizontal=True)
    clients = st.selectbox("固定逻辑客户端数", [2, 3, 4], index=1) if scenario != "集中式" else 1
    if clients > 1:
        st.info(f"按匿名病例索引固定轮转到 {clients} 个逻辑客户端；分类无病例标识时按图像划分。仅评分汇总，不训练或聚合权重。划分版本将随配置保存。")
    head = st.selectbox("输出头与任务信息条件", ["shared", "task-specific"], format_func=lambda x: "共享输出头 / 全局类别编码" if x == "shared" else "任务指定输出头 / 已知任务 ID")
    unseen_key = f"evaluate-unseen-{benchmark_id}"
    unseen_enabled = b["allow_unseen"] and head == "shared"
    if unseen_key not in st.session_state or not unseen_enabled:
        st.session_state[unseen_key] = False
    unseen = st.checkbox("同时评测未见任务", disabled=not unseen_enabled, key=unseen_key)
    if not b["allow_unseen"]:
        st.caption("当前协议不允许对未见任务评分；不会补造前向迁移曲线。")
    st.subheader("02 / 模型或预测")
    method = st.text_input("方法 / 本次评测名称", value="", placeholder="例如：方法名称 · final / seed42", max_chars=80)
    mode_label = st.radio("提交类型", ["预测文件", "已支持结构的模型权重"], horizontal=True)
    mode = "predictions" if mode_label == "预测文件" else "model"
    if b["synthetic"]:
        provenance = "synthetic"
        st.caption("结果来源：合成工程验收；由协议结构确定，不能改标为真实结果。")
    else:
        provenance_labels = {
            "external_predictions_unknown": "外部预测 / 来源未知",
            "untrained_baseline": "未训练工程基线",
            "trained_model_declared": "已训练模型（提交者声明）",
        }
        provenance = st.selectbox("结果来源声明", list(provenance_labels), format_func=provenance_labels.get,
                                  key=f"provenance-{benchmark_id}")
        st.caption("来源类别是提交者声明；平台验证测试评分，不验证训练过程。")
    architecture = None
    model_ok = True
    if mode == "model":
        model_ok = sandbox_available()
        architecture = st.selectbox("已审核模型结构", list(ARCHITECTURES[b["kind"]]), format_func=ARCHITECTURES[b["kind"]].get)
        st.caption("只读 float32 safetensors。仅支持上列结构；UNet / EfficientNet / 自定义模型请先在本地生成预测。")
        if head != "shared":
            st.warning("当前已审核模型仅支持共享输出头。任务指定输出头请使用预测文件提交。")
        if model_ok:
            st.success("本机模型隔离检查通过：无网络、无密钥、输入不含标签；限时限内存。")
        else:
            st.warning("本机未满足模型隔离条件，模型提交关闭；预测评分仍可使用。")
    else:
        st.caption("JSON 或 NPZ/ZIP。数组包使用 task__ids.npy 与 task__pred.npy；客户端由平台固定映射，无需单独上传客户端文件。")
    scope = st.radio("可提供的阶段", ["仅最终阶段", "多个 / 部分阶段"], horizontal=True)
    stages = [len(order)] if scope == "仅最终阶段" else st.multiselect("已有阶段位置", list(range(1, len(order) + 1)), default=[len(order)])
    st.caption("缺少阶段将保留为空；BWT、遗忘或迁移指标只有满足对应条件才计算。")
    st.info("提交后由独立评分服务读取冻结测试集，计算任务主指标及 Final average、BWT、Forgetting、FWT、BWTR；条件不足的指标明确显示为不可计算。")
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
    st.subheader("03 / 检查并开始评测")
    if clients > 1 and mode == "predictions":
        st.caption("预测模式由提交者声明同一阶段各客户端预测来自同一全局模型；平台不要求训练日志，也不伪称已验证模型来源。")
    if not worker_alive():
        st.warning("worker 当前离线；请先由管理员启动服务。不会在网页请求里执行长评测。")
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
            st.success("已提交。配置已冻结，独立 worker 将完成评分。")
            st.button("查看此次评测", on_click=navigate, args=("评测记录",), type="primary")


def preview_arrays(job_id, reference):
    return load_preview(job_dir(job_id), reference)


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
    st.caption(f"{b['title']} · {b['version']} · {config['mode']} · 配置已冻结" + (f" · {supervision}分割" if b["kind"] == "segmentation" else ""))
    timeline(config["order"], b)
    st.caption(config["conditions"])
    summary = result["continual"]["global"]
    columns = st.columns(5)
    labels = {"Final average": "最终任务宏平均", "BWT": "后向迁移 BWT", "Forgetting": "遗忘", "FWT": "前向迁移 FWT", "BWTR": "相对后向迁移 BWTR"}
    for col, metric in zip(columns, labels):
        col.metric(labels[metric], score_text(summary[metric]["value"], b["unit"]))
        col.caption(summary[metric]["reason"])
    st.caption("上述指标由独立评分服务从每阶段测试矩阵计算；— 表示条件不足，不用 0 补齐。")
    detail_label = "类别名称" if b["kind"] == "classification" else "病例结果与可视化"
    tab_matrix, tab_clients, tab_cases, tab_protocol = st.tabs(["阶段—任务矩阵", "逻辑客户端", detail_label, "协议与导出"])
    with tab_matrix:
        client_id = st.selectbox("查看结果层级", list(result["matrices"]), format_func=lambda x: "全体测试样本" if x == "global" else f"逻辑客户端 {x}")
        heatmap(result["matrices"][client_id], config["order"], [f"阶段 {s}" for s in range(1, len(config["order"]) + 1)], b["direction"], b["metric"] + (" ↓" if b["direction"] == "lower" else " ↑"))
        st.caption("— 表示未提交阶段、协议不允许评测或该客户端无样本；不是 0。任务 ID 与阶段位置分别保存。")
        detail = pd.DataFrame([{"指标": k, "值": score_text(v["value"], b["unit"]), "条件 / 原因": v["reason"]} for k, v in result["continual"][client_id].items()])
        st.dataframe(detail, hide_index=True, width="stretch")
    with tab_clients:
        if config["clients"] == 1:
            st.info("本次为集中式评测。新建评测时可选择固定逻辑客户端场景。")
        st.caption(config["client_split"]["source"])
        stage = st.select_slider("查看阶段", options=list(range(1, len(config["order"]) + 1)), value=max(config["stages"]))
        client_ids = [x for x in result["matrices"] if x != "global"]
        heatmap([result["matrices"][c][stage - 1] for c in client_ids], config["order"], client_ids, b["direction"], "客户端 × 测试任务")
        stats = [r for r in result["federated"] if r["stage"] == stage]
        if stats:
            frame = pd.DataFrame(stats).rename(columns={"task_id": "任务", "client_macro": "客户端宏平均", "sample_weighted_accuracy": "按样本数加权准确率", "worst_client": "最低性能客户端", "client_std": "客户端标准差", "client_gap": "客户端极差", "available_clients": "有样本客户端", "total_clients": "客户端总数", "stage": "阶段"})
            st.dataframe(frame, hide_index=True, width="stretch")
            st.caption("分类加权项使用每客户端样本准确率 × 样本数；分割和配准不套用分类加权准确率。TRE 越低越好，因此最低性能客户端对应最大 TRE。")
            dist = pd.DataFrame([r for r in result.get("distributions", []) if r.get("stage") == stage])
            if dist.empty:
                st.info("此阶段没有可展示的客户端样本量。")
            else:
                chart_data = dist.rename(columns={"client_id": "逻辑客户端", "n_samples": "样本数", "task_id": "任务"})
                chart = alt.Chart(chart_data).mark_bar().encode(x="逻辑客户端:N", y="样本数:Q",
                    color=alt.Color("任务:N", scale=alt.Scale(range=["#486b81", "#6d9380", "#8796a5", "#a58b5b", "#627e73", "#a7bbc6"])),
                    tooltip=["逻辑客户端", "任务", "样本数"]).properties(height=250)
                st.altair_chart(chart, width="stretch")
                st.dataframe(dist, hide_index=True, width="stretch")
        else:
            st.info("此阶段未提交，没有可展示的客户端统计。")
    with tab_cases:
        stage = st.selectbox("可视化阶段", sorted(config["stages"]), index=len(config["stages"]) - 1, key=f"case-stage-{job['id']}")
        cases_result = result.get("cases", []) if b["kind"] != "classification" else []
        available_tasks = [tid for tid in config["order"] if any(c["task_id"] == tid and c["stage"] == stage for c in cases_result)]
        task_id = st.selectbox("类别任务" if b["kind"] == "classification" else "病例任务", available_tasks or config["order"], key=f"case-task-{job['id']}")
        cases = [c for c in cases_result if c["task_id"] == task_id and c["stage"] == stage]
        task = next(t for t in b["tasks"] if t["id"] == task_id)
        if b["kind"] == "classification":
            registered = lookup.get(b["id"], {})
            class_names = b.get("class_names") or (registered.get("class_names", {}) if registered.get("version") == b.get("version") else {})
            st.subheader("类别名称")
            st.dataframe(pd.DataFrame([{"Class ID": value, "类别名称": class_names.get(str(value), f"类别 {value}")} for value in task.get("classes", [])]), hide_index=True, width="stretch")
            st.caption("分类只发布任务与逻辑客户端聚合准确率；不显示或下载逐样本正误、样本 ID、病例 ID 或隐藏标签频数。")
        previews = [v for v in result.get("visualizations", []) if v["task_id"] == task_id and v["stage"] == stage]
        if b["kind"] in ("segmentation", "registration"):
            st.subheader("预测结果可视化")
            if previews:
                preview = previews[st.selectbox("选择预览病例", range(len(previews)), format_func=lambda i: f"{previews[i]['case_id']} · {b['metric']} {score_text(previews[i]['score'], b['unit'])}", key=f"preview-{job['id']}-{stage}-{task_id}")]
                try:
                    arrays = preview_arrays(job["id"], preview)
                    if b["kind"] == "segmentation":
                        original, overlay = arrays["original"], arrays["overlay"]
                        left, right = st.columns(2)
                        left.image(original, caption="原始测试切片", width="stretch")
                        right.image(overlay, caption="预测遮罩叠加（绿色）", width="stretch")
                        st.metric("该病例前景 Dice", score_text(preview["score"]))
                        st.caption("预览切片在该病例中按预测前景量固定选取；图中不显示、不导出隐藏测试真值。")
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
                        st.caption("显示移动点到预测配准点的 XY 投影；隐藏固定点只用于服务端 TRE 评分，不进入预览文件。")
                except (OSError, ValueError, KeyError):
                    st.warning("私有可视化文件不可读；数值评分仍以已保存结果为准。")
            else:
                st.info("该记录没有可视化预览；新建的分割 / 配准评测会生成最多 6 个私有病例预览。")
        if b["kind"] != "classification":
            st.subheader("病例数值")
            if cases:
                st.dataframe(pd.DataFrame(cases).drop(columns=["case_index"], errors="ignore"), hide_index=True, width="stretch", height=340)
            else:
                st.info("该任务在选定阶段没有病例结果。")
    with tab_protocol:
        st.json(config)
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
    title("评测记录", "仅显示真实队列与评分状态；失败原因可查询，配置与输入保留。")
    jobs = list_jobs()
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
    title("方法比较", "仅比较基准版本、任务顺序、输出头和评价条件相容的结果；不同任务类型不合成总分。")
    jobs = [j for j in list_jobs() if j["status"] == "completed"]
    if len(jobs) < 2:
        st.info("至少需要两条已完成且协议相容的评测。不会使用示意分数填充对比。")
        return
    mapping = {j["id"]: j for j in jobs}
    selected = st.multiselect("选择 2–4 条已完成评测", list(mapping), max_selections=4,
                             format_func=lambda jid: f"{mapping[jid]['config']['method']} · {mapping[jid]['config']['benchmark']['title']} · {jid[:8]}")
    if len(selected) < 2:
        return
    if len({compatibility(mapping[jid]["config"]) for jid in selected}) != 1:
        st.error("比较已拒绝：基准版本、任务顺序、测试资产、输出头或客户端条件不相容。请选择相同评价条件的记录。")
        return
    b = mapping[selected[0]]["config"]["benchmark"]
    if b.get("synthetic"):
        st.warning("合成验收对比，不代表真实方法性能。")
    rows = []
    for jid in selected:
        result = get_job(jid)["result"]
        for tid, value in zip(result["config"]["order"], result["matrices"]["global"][-1]):
            rows.append({"方法": f"{result['config']['method']} · {jid[:6]}", "任务": tid, b["metric"]: value})
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, width="stretch")
    chart = alt.Chart(df).mark_bar().encode(x=alt.X("任务:N", axis=alt.Axis(labelAngle=0)), y=alt.Y(f"{b['metric']}:Q"),
                                            color=alt.Color("方法:N", scale=alt.Scale(range=["#486b81", "#6d9380", "#8796a5", "#a58b5b"])),
                                            xOffset="方法:N", tooltip=list(df.columns)).properties(height=320)
    st.altair_chart(chart, width="stretch")
    st.caption("比较最终阶段各任务；缺少最终阶段的记录保留为空，不拿最后可见阶段代替最终阶段。")


{"任务中心": task_center, "评测记录": records, "方法比较": compare}[page]()
