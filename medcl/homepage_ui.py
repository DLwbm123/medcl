"""MedCL home presentation; no dataset, inference, worker, or browser-script access.

All buttons below are native Streamlit widgets wired to app.py callbacks. Only
trusted, local illustration assets are embedded; they are never run results.
The module intentionally takes ``st`` and the read-only view data as arguments.
"""
from __future__ import annotations

import base64
from collections import Counter
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

_ASSET_DIR = Path(__file__).resolve().parent / "ui_assets" / "homepage"
_ASSETS = frozenset({"medical-background.webp", "illustration-slice.webp",
                     "illustration-overlay.webp", "illustration-view.webp",
                     "illustration-volume.webp"})

# Original local outline icons. Rendering as <img> avoids depending on inline
# SVG handling in Streamlit's HTML sanitizer or on remote icon fonts.
_ICON_PATHS = {
    "brain": '<path d="M12 3v18M12 5C10 1 5 2 5 6 1 6 1 12 4 13c-2 4 1 7 4 6 1 3 4 2 4-1M12 5c2-4 7-3 7 1 4 0 4 6 1 7 2 4-1 7-4 6-1 3-4 2-4-1M5 6c0 2 1 3 3 3M4 13c2-1 4 0 4 2M19 6c0 2-1 3-3 3M20 13c-2-1-4 0-4 2M8 5v1m8-1v1M8 18v-1m8 1v-1"/>',
    "home": '<path d="m3 10 9-7 9 7M5 9v12h5v-7h4v7h5V9"/>',
    "document": '<path d="M14 3H5v18h14V8zM14 3v5h5M8 12h8M8 16h6"/>',
    "bars": '<path d="M4 20V10m6 10V4m6 16v-8m4 8H2"/>',
    "compare": '<path d="M4 20V9h4v11m4 0V4h4v16m4 0v-8M2 21h20"/>',
    "cube": '<path d="m12 3 9 5v9l-9 5-9-5V8zM3 8l9 5 9-5M12 13v9M7.5 5.5l9 5"/>',
    "list": '<path d="M9 6h11M9 12h11M9 18h11"/><rect x="3" y="5" width="2" height="2" rx=".2"/><rect x="3" y="11" width="2" height="2" rx=".2"/><rect x="3" y="17" width="2" height="2" rx=".2"/>',
    "target": '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><path d="M12 1v5m0 12v5M1 12h5m12 0h5"/>',
    "database": '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v7c0 4 16 4 16 0V5M4 12v7c0 4 16 4 16 0v-7"/>',
    "shield": '<path d="m12 3 9 3v6c0 5-5 9-9 11-4-2-9-6-9-11V6z"/><path d="m8 12 3 3 5-6"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v6l4 2"/>',
    "arrow": '<path d="M4 12h16m-6-6 6 6-6 6"/>',
    "chevron": '<path d="m9 5 7 7-7 7"/>',
    "empty": '<path d="M3 16 6 5h12l3 11v5H3zM3 16h5l2 3h4l2-3h5"/>',
}


@lru_cache(maxsize=64)
def icon_uri(name: str, color: str = "#355F8A") -> str:
    if name not in _ICON_PATHS:
        raise ValueError(f"Unknown local icon: {name}")
    if len(color) != 7 or color[0] != "#" or any(c not in "0123456789abcdefABCDEF" for c in color[1:]):
        raise ValueError("Icon color must be a six-digit hex value")
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
           f'fill="none" stroke="{color}" stroke-width="1.65" '
           'stroke-linecap="round" stroke-linejoin="round">' + _ICON_PATHS[name] + '</svg>')
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode("ascii")


def icon(name: str, color: str = "#355F8A", size: int = 24) -> str:
    if not isinstance(size, int) or not 8 <= size <= 128:
        raise ValueError("Invalid icon size")
    return (f'<img class="medcl-icon" src="{icon_uri(name, color)}" width="{size}" '
            f'height="{size}" alt="" aria-hidden="true" draggable="false">')


@lru_cache(maxsize=8)
def asset_uri(filename: str) -> str:
    if filename not in _ASSETS:
        raise ValueError("Only bundled presentation assets may be read")
    content = (_ASSET_DIR / filename).read_bytes()
    if not content.startswith(b"RIFF") or content[8:12] != b"WEBP":
        raise ValueError("Invalid bundled WebP asset")
    return "data:image/webp;base64," + base64.b64encode(content).decode("ascii")


@lru_cache(maxsize=1)
def stylesheet() -> str:
    css = (_ASSET_DIR / "homepage.css").read_text(encoding="utf-8")
    css = css.replace("__MEDICAL_BACKGROUND__", asset_uri("medical-background.webp"))
    for name in ("home", "document", "bars", "compare", "arrow", "database", "chevron"):
        css = css.replace(f"__ICON_{name.upper()}__", icon_uri(name))
    css = css.replace("__ICON_ARROW_WHITE__", icon_uri("arrow", "#FFFFFF"))
    return css


def install_styles(st: Any) -> None:
    """Install after app.py's base theme. No static server or remote resource."""
    st.html("<style>" + stylesheet() + "</style>")


def render_header(st: Any, nav: Sequence[str], control: Callable[..., str]) -> str:
    with st.container(key="topbar"):
        brand_column, nav_column = st.columns([1.08, 1], vertical_alignment="center")
        with brand_column:
            st.html('<div class="medcl-brand">' + icon("brain", "#30629F", 32) +
                    '<strong>MedCL</strong><span>医学影像持续学习评测平台</span></div>')
        with nav_column:
            # Preserve the existing required-selection guard and session key.
            return control("主导航", nav, "nav", "collapsed")


def hero_copy_html() -> str:
    return '''<div class="medcl-hero-copy">
      <p class="medcl-eyebrow">CONTINUAL LEARNING FOR MEDICAL IMAGING</p>
      <h1 class="medcl-hero-title">MedCL</h1>
      <h2 class="medcl-hero-subtitle">医学影像持续学习评测平台</h2>
      <p class="medcl-hero-description">面向分割、分类与配准任务，统一整理持续学习评测结果，<br class="medcl-desktop-break">分析阶段表现、任务差异与病例可视化。</p>
    </div>'''


def capabilities_html() -> str:
    items = (("document", "多种任务协议", "明确任务与评测条件"),
             ("bars", "统一评测指标", "查看阶段与任务表现"),
             ("cube", "可视化分析", "结果矩阵与病例预览"),
             ("shield", "评测记录", "保存配置与评分结果"))
    return '<div class="medcl-capabilities">' + ''.join(
        '<div class="medcl-capability">' + icon(i, "#38659A", 27) +
        f'<div><strong>{title}</strong><span>{detail}</span></div></div>'
        for i, title, detail in items) + '</div>'


def hero_visual_html() -> str:
    cells = ['<span></span>'] + [f'<span>任务 {i}</span>' for i in range(1, 5)]
    shades = ("#477EB7", "#6F99C3", "#92B1D0", "#BDD1E5")
    for stage in range(4):
        cells.append(f'<span>阶段 {stage + 1}</span>')
        for task in range(4):
            if task > stage:
                cells.append('<i class="medcl-matrix-cell is-missing" aria-label="未评测">—</i>')
            else:
                cells.append(f'<i class="medcl-matrix-cell" style="background:{shades[stage-task]}"></i>')
    tiles = (("illustration-slice.webp", "影像示意"),
             ("illustration-overlay.webp", "预测叠加"),
             ("illustration-view.webp", "切片预览"),
             ("illustration-volume.webp", "三维预览"))
    images = ''.join(f'<figure><img src="{asset_uri(file)}" '
                     f'alt="AI 生成的{label}装饰素材，非真实病例" width="240" height="240">'
                     f'<figcaption>{label}</figcaption></figure>' for file, label in tiles)
    return ('<div class="medcl-hero-visual" aria-label="功能示意，非实验结果">'
            '<div class="medcl-matrix-panel"><div class="medcl-demo-heading">'
            '<strong>持续学习结果矩阵</strong><span>阶段 × 任务</span></div>'
            '<div class="medcl-matrix-grid">' + ''.join(cells) + '</div>'
            '<div class="medcl-matrix-note">已见任务评测<span>— 未评测</span></div></div>'
            '<div class="medcl-case-panel"><div class="medcl-demo-heading">'
            '<strong>病例可视化</strong>' + icon("arrow", "#849CB7", 16) + '</div>'
            '<div class="medcl-case-grid">' + images + '</div>'
            '<p class="medcl-demo-disclaimer">功能示意 · 非实验结果 · 非真实病例</p></div></div>')


def overview_html(counts: Sequence[tuple[str, int, str, str]]) -> str:
    output = ['<div class="medcl-overview-grid">']
    for label, value, image, tone in counts:
        if type(value) is not int or value < 0 or tone not in ("blue", "green", "purple", "gold"):
            raise ValueError("Invalid overview value")
        colors = {"blue": "#326FB1", "green": "#259B7F", "purple": "#835AC0", "gold": "#A5822E"}
        output.append(f'<div class="medcl-stat {tone}">{icon(image, colors[tone], 26)}'
                      f'<div><strong>{value}</strong><span>{escape(label)}</span></div></div>')
    return ''.join(output) + '</div>'


def _section_html(title: str, image: str | None = None, detail: str = "") -> str:
    return ('<div class="medcl-section-heading">' + (icon(image, size=20) if image else '') +
            f'<h2>{escape(title)}</h2></div>' +
            (f'<p class="medcl-section-description">{escape(detail)}</p>' if detail else ''))


def render_homepage(
    st: Any, *, benchmarks: Sequence[Mapping[str, Any]], jobs: Sequence[Mapping[str, Any]],
    readiness: Callable[..., tuple[bool, str]], kinds: Mapping[str, str], statuses: Mapping[str, str],
    on_start: Callable[[], None], on_records: Callable[[], None],
    on_task: Callable[[str], None], on_job: Callable[[str], None],
    on_showcase: Callable[[str], None] | None = None, show_demos: bool = False,
) -> None:
    """Render using existing read-only catalog and visible-jobs snapshots."""
    available = Counter()
    for benchmark in benchmarks:
        if not benchmark.get("synthetic") and readiness(benchmark)[0]:
            available[benchmark["kind"]] += 1
    visible = sorted(jobs, key=lambda job: str(job.get("created_at", "")), reverse=True)
    with st.container(key="medcl-home-hero"):
        copy_column, visual_column = st.columns([1.13, 1], gap="large", vertical_alignment="center")
        with copy_column:
            st.html(hero_copy_html())
            with st.container(key="medcl-home-actions"):
                first, second, _ = st.columns([1, 1.18, .9], gap="small")
                first.button("开始评测", key="medcl-home-start", on_click=on_start,
                             type="primary", width="stretch")
                second.button("查看评测记录", key="medcl-home-records", on_click=on_records,
                              width="stretch")
            st.html(capabilities_html())
        with visual_column:
            st.html(hero_visual_html())

    with st.container(key="medcl-home-tasks"):
        st.html(_section_html("选择评测任务", detail="选择任务类型，浏览可用协议并开始评测。"))
        cols = st.columns(3, gap="medium")
        definitions = (
            ("segmentation", "医学影像分割", "器官与病灶分割的持续学习评测", "cube"),
            ("classification", "医学影像分类", "医学影像分类任务的持续学习评测", "list"),
            ("registration", "医学影像配准", "任务增量 · 影像配准持续学习", "target"),
        )
        for col, (kind, name, description, glyph) in zip(cols, definitions):
            with col:
                with st.container(border=True, key=f"medcl-home-task-{kind}"):
                    description_col, button_col = st.columns([7, 1], gap="small", vertical_alignment="center")
                    state = f"{available[kind]} 个可用真实协议" if available[kind] else "自动评分数据待接入"
                    with description_col:
                        st.html(f'<div class="medcl-task-heading"><span class="medcl-task-symbol {kind}">'
                                + icon(glyph, "#FFFFFF", 26) + '</span><div>'
                                f'<h3>{name}</h3><p>{description}</p>'
                                f'<span class="medcl-task-state">{state}</span></div></div>')
                    button_col.button(f"进入{kinds[kind]}任务", key=f"home-enter-{kind}",
                                      on_click=on_task, args=(kind,), width="stretch",
                                      help=f"{name} · {state}")
                    if on_showcase is not None:
                        examples = (("segmentation-full", "全监督"), ("segmentation-weak", "弱监督")) if kind == "segmentation" else ((kind, f"{kinds[kind]}"),)
                        for example, label in examples:
                            st.button(label, key=f"home-showcase-{example}", on_click=on_showcase,
                                      args=(example,), width="stretch")

    with st.container(key="medcl-home-lower"):
        recent, overview = st.columns([1.2, 1], gap="medium")
        with recent:
            with st.container(border=True, key="medcl-home-recent"):
                heading, all_records = st.columns([4, 1], vertical_alignment="center")
                heading.html(_section_html("最近评测记录", "document"))
                all_records.button("查看全部 →", key="medcl-home-all-records", on_click=on_records,
                                   width="stretch")
                if not visible:
                    st.html('<div class="medcl-empty">' + icon("empty", "#94A8BE", 35) +
                            '<div><strong>还没有评测记录</strong><span>选择上方任务，开始第一条评测。</span></div></div>')
                else:
                    st.html('<div class="medcl-recent-head"><span>时间 · UTC</span><span>协议 / 方法</span>'
                            '<span>状态</span><span>操作</span></div>')
                    for job in visible[:3]:
                        cfg = job["config"]
                        protocol = cfg["benchmark"]
                        stamp = str(job["created_at"]).replace("T", " ").replace("+00:00", "")[:16]
                        state = str(job.get("status", "unknown"))
                        status_class = state if state in {"completed", "queued", "running", "failed"} else "unknown"
                        info, action = st.columns([11, 1], gap="small", vertical_alignment="center")
                        info.html('<div class="medcl-recent-row">'
                                  f'<time>{escape(stamp)}</time><div class="medcl-recent-details">'
                                  f'<strong title="{escape(str(protocol["title"]), quote=True)}">{escape(str(protocol["title"]))}</strong>'
                                  f'<span title="{escape(str(cfg["method"]), quote=True)}">{escape(str(cfg["method"]))}</span></div>'
                                  f'<span class="medcl-status {status_class}">{escape(statuses.get(state, "未知状态"))}</span></div>')
                        action.button("查看", key=f"home-job-{job['id']}", on_click=on_job,
                                      args=(job["id"],), width="stretch")
        with overview:
            with st.container(border=True, key="medcl-home-overview"):
                st.html(_section_html("平台数据概览", "bars"))
                registered = {b["kind"] for b in benchmarks if not b.get("synthetic")}
                values = (("可用真实协议", sum(available.values()), "cube", "blue"),
                          ("已登记任务类型", len(registered), "database", "green"),
                          ("已完成评测", sum(j["status"] == "completed" for j in visible), "bars", "purple"),
                          ("排队 / 运行中", sum(j["status"] in ("queued", "running") for j in visible), "clock", "gold"))
                st.html(overview_html(values))
                st.html('<p class="medcl-overview-note">按当前协议和可见记录汇总；登记不代表真实数据已接入。</p>')
                if show_demos:
                    st.caption("开发模式：记录统计包含当前可见的合成验收记录。")
    st.html('<div class="medcl-home-footer"><span>本地研究评测 · 非临床用途</span>'
            '<span>MedCL · 医学影像持续学习</span></div>')
