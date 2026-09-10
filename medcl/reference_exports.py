"""Self-contained historical reports; never route them through platform scoring."""
import base64
import csv
from html import escape
import io
import json
import os
from functools import lru_cache
from pathlib import Path
import shutil
import subprocess

from medcl.reference_results import METRICS, NOTICE, PROTOCOL


@lru_cache(maxsize=1)
def local_export_font():
    """Require a local CJK font, instead of silently exporting missing glyphs."""
    from PIL import ImageFont
    candidates = [Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
                  Path("/System/Library/Fonts/Hiragino Sans GB.ttc"), Path("C:/Windows/Fonts/msyh.ttc")]
    configured = os.environ.get("MEDCL_EXPORT_FONT")
    if configured:
        candidates = [Path(configured).expanduser()]
    if not configured and shutil.which("fc-match"):
        match = subprocess.run(["fc-match", "-f", "%{file}", "sans-serif:lang=zh-cn"], capture_output=True, text=True, timeout=5)
        if match.returncode == 0 and match.stdout:
            candidates.append(Path(match.stdout))
    for path in candidates:
        if path.is_file():
            font = ImageFont.truetype(str(path), 20)
            # Missing characters share a fallback box; distinct masks verify CJK support.
            if bytes(font.getmask("医")) != bytes(font.getmask("学")):
                return path, font.getname()[0]
    raise RuntimeError("未找到可用的本地中文字体。请安装系统 CJK 字体后重试；本功能不会下载或分发论文字体。")


def json_bytes(payload):
    return (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def csv_bytes(records):
    output = io.StringIO(newline="")
    fields = list(dict.fromkeys(k for r in records for k in r)) or ["scenario", "method_id", "metric", "mean", "sd", "status", "missing_reason"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for record in records:
        safe = {}
        for key, value in record.items():
            if isinstance(value, (list, dict)):
                value = json.dumps(value, ensure_ascii=False, allow_nan=False)
            if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")):
                value = "'" + value
            safe[key] = value
        writer.writerow(safe)
    return output.getvalue().encode("utf-8-sig")


def render_chart(chart, width=880):
    # Lazy import keeps the original platform comparison usable without export extras.
    try:
        import vl_convert as vlc
    except ImportError as exc:
        raise RuntimeError("图表导出需要 requirements.txt 中的 vl-convert-python。") from exc
    spec = chart.to_dict()
    spec["width"] = width
    font_path, family = local_export_font()
    vlc.register_font_directory(str(font_path.parent))
    spec.setdefault("config", {})["font"] = family
    # Offline renderer: no external datasets, CDN or font downloads.
    svg = vlc.vegalite_to_svg(spec, vl_version="6.4", allowed_base_urls=[])
    png = vlc.vegalite_to_png(spec, vl_version="6.4", scale=1.5, allowed_base_urls=[])
    if len(svg) < 500 or not png.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("图表导出为空或无效")
    return {"svg": svg.encode("utf-8"), "png": png}


def export_view(payload, charts, pictures=()):
    rendered = {name: render_chart(chart) for name, chart in charts.items()}
    parts = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>",
             "<title>第三章持续分割 Benchmark</title><style>body{font-family:system-ui,'PingFang SC',sans-serif;background:#F5F8FC;color:#172B43;max-width:1100px;margin:auto;padding:24px}section{background:white;border:1px solid #DDE5EE;border-radius:12px;padding:20px;margin:18px 0}img{width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}h1{font-size:28px}p{line-height:1.65}</style>",
             f"<h1>第三章持续分割 Benchmark</h1><p>{NOTICE}</p>",
             f"<p>当前范围：{escape(json.dumps(payload['scope'], ensure_ascii=False))}</p>",
             "<p>误差线仅为论文报告 SD，重复次数与主表变异来源未知；均值差不代表统计显著性。资源前沿只含当前筛选 CL 方法，按报告均值计算。</p>"]
    for name, item in rendered.items():
        # PNG embeds ensure offline Chinese readability even without the source font.
        parts.append(f"<section><h2>{escape(name)}</h2><img alt='{escape(name)}' src='data:image/png;base64,{base64.b64encode(item['png']).decode()}'></section>")
    for path, caption in pictures:
        parts.append(f"<section><img alt='{escape(caption)}' src='data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}'><p>{escape(caption)}</p></section>")
    parts.append("<section><h2>指标、协议与数据边界</h2>")
    for key, explanation in {**METRICS, **PROTOCOL}.items():
        parts.append(f"<p><b>{escape(key)}</b>：{escape(explanation)}</p>")
    parts.append("</section><section><h2>数值、缺失原因与来源</h2><pre>" + escape(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)) + "</pre></section></html>")
    return {"json": json_bytes(payload), "csv": csv_bytes(payload["records"]),
            "html": "\n".join(parts).encode("utf-8"), "charts": rendered}
