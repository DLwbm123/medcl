"""Pure Altair factories for reported historical values (no job queries)."""
from collections import Counter
import altair as alt

from medcl.reference_results import COLORS, FAMILIES, SCENARIOS, pareto, sam_cells, value

FONT = "PingFang SC"
COLOR_SCALE = alt.Scale(domain=list(FAMILIES), range=list(COLORS.values()))
CATEGORY_LEGEND = alt.Legend(labelExpr=str(FAMILIES).replace("'", '"') + "[datum.label]")
COLOR = alt.Color("family:N", title="方法类别", scale=COLOR_SCALE,
                  legend=CATEGORY_LEGEND)
SHAPE = alt.Shape("family:N", title="方法类别", scale=alt.Scale(domain=list(FAMILIES), range=["circle", "square", "diamond", "triangle-up", "cross"]), legend=CATEGORY_LEGEND)
TOOLTIP = [alt.Tooltip("method_id:N", title="方法"), alt.Tooltip("category:N", title="类别"),
           alt.Tooltip("mean:Q", title="原文均值", format=".3f"), alt.Tooltip("sd:Q", title="原文 SD", format=".3f"),
           alt.Tooltip("source:N", title="来源"), alt.Tooltip("source_type:N", title="研究类型")]


def data(rows):
    return alt.Data(values=rows)


def finish(chart, title, height=330):
    return chart.properties(title=title, width="container", height=height + 110,
        padding={"top": 35, "left": 12, "right": 48, "bottom": 18},
        autosize=alt.AutoSizeParams(type="fit", contains="padding")).configure(
        font=FONT, background="white",
        title={"fontSize": 17, "anchor": "start", "color": "#172B43", "offset": 18},
        axis={"labelFontSize": 12, "labelLimit": 220, "titleFontSize": 12, "domain": False, "gridColor": "#E9EFF5", "labelColor": "#475467", "titleColor": "#344054"},
        legend={"orient": "bottom", "labelFontSize": 12, "titleFontSize": 12, "columns": 2},
        view={"stroke": None})


def source(row):
    position = f"L{row['source_line']}" if row.get("source_line") else f"p{row.get('source_page', 1)}"
    return f"{row['source_file']} · {row['source_label']} · {position}"


def metric_rows(rows, metric):
    values = []
    for r in rows:
        v = r["metrics"].get(metric, {})
        if v.get("mean") is None:
            continue
        values.append(dict(method_id=r["method_id"], display=r["method_id"] + (" · 参照" if r["role"] == "reference" else ""),
            family=r["family"], category=FAMILIES[r["family"]], mean=v["mean"], sd=v["sd"],
            low=v["mean"] - v["sd"] if v["sd"] is not None else None,
            high=v["mean"] + v["sd"] if v["sd"] is not None else None,
            label=f"{v['mean']:.3f}", source=source(r), source_type=r["source_type"]))
    return sorted(values, key=lambda r: (-r["mean"], r["method_id"]))


def ranking(rows, metric="A-Dice", sd=True):
    values = metric_rows(rows, metric)
    bar = metric in ("A-Dice", "WCD")
    base = alt.Chart(data(values)).encode(y=alt.Y("display:N", sort=[r["display"] for r in values], title=None,
        axis=alt.Axis(labelLimit=190)), tooltip=TOOLTIP)
    x = alt.X("mean:Q", title=f"{metric} · {'Dice' if metric in ('A-Dice', 'WCD', 'E-FWT') else '比率'}", scale=alt.Scale(zero=bar, nice=True))
    mark = base.mark_bar(cornerRadiusEnd=3, opacity=.85).encode(x=x, color=COLOR) if bar else base.mark_point(filled=True, size=90).encode(x=x, color=COLOR, shape=SHAPE)
    layers = [mark]
    if sd and any(r["sd"] and r["sd"] > 0 for r in values):
        interval = base.transform_filter("datum.sd > 0").mark_rule(color="#33465B", strokeWidth=1.6).encode(x="low:Q", x2="high:Q")
        layers.extend([interval, base.transform_filter("datum.sd > 0").mark_tick(color="#33465B", size=9).encode(x="low:Q"),
                       base.transform_filter("datum.sd > 0").mark_tick(color="#33465B", size=9).encode(x="high:Q")])
    # A visible point preserves true zero even when its bar/interval has zero width.
    if bar and any(r["mean"] == 0 for r in values):
        layers.append(base.transform_filter("datum.mean === 0").mark_point(filled=True, size=55).encode(x=x, color=COLOR))
    layers.append(base.mark_text(align="left", dx=7, dy=-9 if sd else 0, fontSize=11).encode(x=x, text="label:N"))
    if not bar:
        layers.insert(0, alt.Chart(data([{"baseline": 1 if metric == "RMA" else 0}])).mark_rule(strokeDash=[4, 3], color="#536E87").encode(x="baseline:Q"))
    return finish(alt.layer(*layers), f"{metric} · 报告均值" + (" ± SD" if sd else ""), max(250, 32 * len(values)))


def composition(rows):
    counts = Counter(r["family"] for r in rows if r["role"] == "continual_method")
    total = sum(counts.values())
    values = [dict(family=k, count=v, label=f"{FAMILIES[k]} · {v} · {v / total:.1%}") for k, v in counts.items()]
    ring = alt.Chart(data(values)).mark_arc(innerRadius=52, outerRadius=88).encode(
        theta="count:Q", color=alt.Color("label:N", title=None,
            scale=alt.Scale(domain=[v["label"] for v in values], range=[COLORS[v["family"]] for v in values]),
            legend=alt.Legend(columns=1, labelLimit=300)),
        tooltip=[alt.Tooltip("label:N", title="方法数量构成")])
    center = alt.Chart(data([{"text": f"{total} 个 CL 方法"}])).mark_text(fontSize=17, fontWeight=600).encode(text="text:N")
    return finish(ring + center, "方法数量构成", 235)


def tradeoff(rows, resource=None, frontier=True):
    values = []
    xmetric, ymetric = (resource, "A-Dice") if resource else ("BWTR", "RMA")
    for row in rows:
        if value(row, xmetric) is None or value(row, ymetric) is None:
            continue
        values.append(dict(method_id=row["method_id"], family=row["family"], category=FAMILIES[row["family"]],
            x=value(row, xmetric), y=value(row, ymetric), adice=value(row, "A-Dice"),
            source=source(row), drr_star=row["drr_star"], data_access=row["data_access"], source_type=row["source_type"]))
    base = alt.Chart(data(values)).encode(
        x=alt.X("x:Q", title=xmetric, scale=alt.Scale(zero=True if resource else False, padding=14)),
        y=alt.Y("y:Q", title=ymetric, scale=alt.Scale(zero=False, padding=14)),
        tooltip=["method_id:N", "category:N", alt.Tooltip("x:Q", title=xmetric, format=".3f"),
                 alt.Tooltip("y:Q", title=ymetric, format=".3f"), "drr_star:N", "data_access:N", "source:N", "source_type:N"])
    points = base.mark_point(filled=True, opacity=.85).encode(color=COLOR, shape=SHAPE,
        size=alt.Size("adice:Q", title="A-Dice（点面积）", scale=alt.Scale(range=[65, 280], zero=True), legend=None))
    layers = [points]
    if resource and frontier:
        ids = pareto(rows, resource)
        if len(ids) > 1:
            layers.append(base.transform_filter(alt.FieldOneOfPredicate(field="method_id", oneOf=ids)).mark_line(color="#172B43", strokeDash=[5, 4]).encode(order="x:Q"))
        if ids:
            layers.append(base.transform_filter(alt.FieldOneOfPredicate(field="method_id", oneOf=ids)).mark_point(size=340, color="#172B43", strokeWidth=1.5).encode())
    if not resource:
        layers.extend([alt.Chart(data([{"b": 0}])).mark_rule(strokeDash=[4, 4], color="#9BACBC").encode(x="b:Q"),
                       alt.Chart(data([{"b": 1}])).mark_rule(strokeDash=[4, 4], color="#9BACBC").encode(y="b:Q")])
    return finish(alt.layer(*layers), f"{xmetric} × {ymetric}", 340)


def dumbbell(rows, sd=True):
    values, pairs = [], []
    for r in rows:
        if value(r, "WCD") is None:
            continue
        pairs.append(dict(method_id=r["method_id"], a=value(r, "A-Dice"), w=value(r, "WCD")))
        for metric in ("A-Dice", "WCD"):
            cell = metric_rows([r], metric)[0]
            values.append(dict(cell, metric=metric))
    order = [r["method_id"] for r in sorted(pairs, key=lambda r: -r["a"])]
    link = alt.Chart(data(pairs)).mark_rule(color="#C4D1DF").encode(x="a:Q", x2="w:Q", y=alt.Y("method_id:N", sort=order, title=None))
    base = alt.Chart(data(values)).encode(x=alt.X("mean:Q", title="Dice（两种不同汇总对象）", scale=alt.Scale(zero=True)), y=alt.Y("method_id:N", sort=order, title=None), tooltip=TOOLTIP + ["metric:N"])
    points = base.mark_point(filled=True, size=85).encode(color=alt.Color("metric:N", scale=alt.Scale(domain=["A-Dice", "WCD"], range=["#355F8A", "#C97724"])), shape="metric:N")
    layers = [link, points]
    if sd:
        layers.insert(1, base.mark_rule(opacity=.6).encode(x="low:Q", x2="high:Q", color="metric:N"))
    return finish(alt.layer(*layers), "A-Dice / WCD", max(260, len(pairs) * 31))


def cross_scene(rows, methods, metric):
    lookup = {(r["method_id"], r["scenario"]): r for r in rows}
    values = []
    for method in methods:
        for scenario in SCENARIOS:
            row = lookup.get((method, scenario))
            v = value(row, metric) if row else None
            values.append(dict(method_id=method, scenario=scenario, mean=v, text=f"{v:.3f}" if v is not None else "未报告",
                source=source(row) if row else f"tab:benchmark-{scenario.split('-')[0].lower()}-results · 未报告该方法",
                reason=row["metrics"][metric]["missing_reason"] if row and metric in row["metrics"] else "该场景主表没有此方法记录"))
    base = alt.Chart(data(values)).encode(x=alt.X("scenario:N", sort=list(SCENARIOS), title="场景（不是学习阶段）", axis=alt.Axis(labelAngle=0,
        labelExpr="{'Domain-CL':'域增量','Class-CL':'类别增量','Organ-CL':'任务增量'}[datum.label]")),
        y=alt.Y("method_id:N", sort=methods, title=None), tooltip=["method_id:N", "scenario:N", "text:N", "reason:N", "source:N"])
    scale = alt.Scale(scheme="blues", domain=[0, 1]) if metric == "A-Dice" else alt.Scale(scheme="redblue", domainMid=0 if metric == "BWTR" else 1, zero=False)
    tiles = base.mark_rect(stroke="white", strokeWidth=2).encode(color=alt.condition("datum.mean !== null", alt.Color("mean:Q", title=metric, scale=scale), alt.value("#EDF1F6")))
    labels = base.mark_text(fontSize=12).encode(text="text:N", color=alt.condition("datum.mean !== null && datum.mean > 0.55" if metric == "A-Dice" else "false", alt.value("white"), alt.value("#172B43")))
    return finish(tiles + labels, f"跨场景 · {metric} 原始值", max(260, 31 * len(methods)))


def order_sd(records, metric):
    values = [dict(r, family="regularization" if r["method_id"].startswith("Regu") else "replay") for r in records if r["metric"] == metric]
    order = [r["method_id"] for r in sorted(values, key=lambda r: (-r["order_sd"], r["method_id"]))]
    base = alt.Chart(data(values)).encode(y=alt.Y("method_id:N", sort=order, title=None), x=alt.X("order_sd:Q", title="十种任务顺序的 SD", scale=alt.Scale(zero=True)), tooltip=["method_id:N", alt.Tooltip("order_sd:Q", format=".4f"), "n_orders:Q", "source_file:N", "source_page:Q"])
    return finish(base.mark_bar().encode(color=COLOR) + base.mark_text(align="left", dx=5, fontSize=12).encode(text=alt.Text("order_sd:Q", format=".4f")), f"{metric} · 任务顺序波动", 310)


def sam_heatmap(registry, mode):
    values = sam_cells(registry)
    delta = mode == "LoRA−SAM"
    if delta:
        matrices = registry["sam_matrices"]
        values = [dict(r, mean=round(matrices["SAM-LoRA"][r["stage"]-1][int(r["test_task"][1:])-1] - r["mean"], 3), source_type="derived_from_reported_matrix") for r in values if r["method_id"] == "SAM"]
    else:
        values = [r for r in values if r["method_id"] == mode]
    bound = max(abs(r["mean"]) for r in values) if delta else 1
    scale = alt.Scale(scheme="redblue", domain=[-bound, bound], domainMid=0) if delta else alt.Scale(scheme="blues", domain=[0, 1])
    base = alt.Chart(data(values)).encode(x=alt.X("test_task:O", title="测试任务", axis=alt.Axis(labelAngle=0)), y=alt.Y("stage:O", title="完成训练阶段", sort="ascending"), tooltip=["stage:O", "test_task:N", alt.Tooltip("mean:Q", title="Δ Dice" if delta else "Dice", format="+.3f" if delta else ".3f"), "region:N", "source_file:N", "source_page:Q", "source_type:N"])
    rect = base.mark_rect(stroke="white", strokeWidth=2).encode(color=alt.Color("mean:Q", title="Δ Dice" if delta else "Dice", scale=scale))
    text = base.mark_text(fontSize=12).encode(text=alt.Text("mean:Q", format="+.3f" if delta else ".3f"), color=alt.condition(f"abs(datum.mean) > {bound * .7}" if delta else "datum.mean > 0.55", alt.value("white"), alt.value("#172B43")))
    diagonal = base.transform_filter("datum.stage === toNumber(substring(datum.test_task, 1))").mark_rect(fillOpacity=0, stroke="#172B43", strokeWidth=2).encode()
    return finish(rect + text + diagonal, mode, 280)


def sam_curve(registry, task):
    values = [r for r in sam_cells(registry) if r["test_task"] == task]
    curve = alt.Chart(data(values)).mark_line(point=True, interpolate="linear", strokeWidth=2).encode(
        x=alt.X("stage:Q", title="完成训练阶段", axis=alt.Axis(values=[1, 2, 3, 4, 5, 6])),
        y=alt.Y("mean:Q", title="Dice", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("method_id:N", title="适配方式", scale=alt.Scale(domain=["SAM", "SAM-LoRA"], range=["#355F8A", "#C97724"])),
        strokeDash="method_id:N", tooltip=["method_id:N", "stage:Q", "test_task:N", "mean:Q", "region:N", "source_file:N"])
    first = alt.Chart(data([{"stage": int(task[1:])}])).mark_rule(strokeDash=[5, 3], color="#687A8D").encode(x="stage:Q")
    return finish(curve + first, f"{task} · 六阶段实际观测", 320)
