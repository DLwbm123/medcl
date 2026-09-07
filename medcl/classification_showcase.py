"""Presentation for the private classification gallery; no inference or scoring."""
import base64
from html import escape
from io import BytesIO

from PIL import Image

DATASETS = {
    "pathmnist": {"name": "PathMNIST", "modality": "组织病理", "en": "HISTOPATHOLOGY",
                  "size": 128, "classes": 9, "file": "classification", "accent": "#7662a4",
                  "description": "结直肠组织切片，浏览不同组织的形态与纹理。",
                  "labels": ("脂肪组织", "背景", "组织碎屑", "淋巴细胞", "黏液", "平滑肌", "正常结肠黏膜", "癌相关间质", "结直肠腺癌上皮")},
    "skin": {"name": "Skin Cancer", "modality": "皮肤影像", "en": "SKIN IMAGING",
             "size": 128, "classes": 6, "file": "classification-skin", "accent": "#b46e4b",
             "description": "皮肤病变图像，浏览病变区域的颜色、边界与表面纹理。",
             # The prepared six-class arrays have numeric labels but no verified name mapping.
             "labels": tuple(f"类别 {i}" for i in range(6))},
    "hyperkvasir": {"name": "HyperKvasir", "modality": "消化道内镜", "en": "GASTROINTESTINAL ENDOSCOPY",
                    "size": 224, "classes": 20, "file": "classification-hyperkvasir", "accent": "#367f85",
                    "description": "消化道内镜图像，浏览黏膜、解剖标志与不同检查视野。",
                    # Order verified against the prepared HyperKvasir20 class_mapping.json.
                    "labels": ("BBPS 2–3", "息肉", "盲肠", "染色抬举息肉", "幽门", "染色切除边缘", "Z 线", "胃内反转视图", "BBPS 0–1",
                               "溃疡性结肠炎 · 2 级", "食管炎 · A 级", "直肠反转视图", "食管炎 · B–D 级", "溃疡性结肠炎 · 1 级",
                               "溃疡性结肠炎 · 3 级", "粪便嵌塞", "短段 Barrett 食管", "Barrett 食管", "溃疡性结肠炎 · 0–1 级", "溃疡性结肠炎 · 2–3 级")},
}


def card_html(dataset, arrays):
    """Embed the validated native pixels, with scoped responsive presentation."""
    info = DATASETS[dataset]
    label = int(arrays["class_id"])
    name, modality, category = (escape(value) for value in (info["name"], info["modality"], info["labels"][label]))
    height, width = arrays["image"].shape[:2]
    buffer = BytesIO()
    Image.fromarray(arrays["image"]).save(buffer, format="PNG")
    pixels = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f'''<style>
    .medcl-class-card {{display:grid;grid-template-columns:minmax(260px,1.05fr) minmax(260px,1fr);border:1px solid #e0e7ef;border-radius:22px;overflow:hidden;background:#fff;box-shadow:0 12px 40px #253d5810;color:#203349}}
    .medcl-class-visual {{padding:26px;background:linear-gradient(145deg,#edf1f7,#f8fafc);display:flex;flex-direction:column;gap:20px}}
    .medcl-class-top {{display:flex;justify-content:space-between;align-items:center;font-size:11px;letter-spacing:1.8px;color:#516780;font-weight:650}}
    .medcl-class-rgb {{border:1px solid #ced8e3;border-radius:6px;padding:4px 8px;letter-spacing:1px}}
    .medcl-class-image {{display:block;width:min(100%,384px);height:auto;aspect-ratio:1;object-fit:contain;margin:auto;border-radius:12px;box-shadow:0 8px 24px #243a5420}}
    .medcl-class-image-note {{display:flex;justify-content:space-between;color:#64758b;font-size:12px;gap:12px}}
    .medcl-class-info {{padding:38px;display:flex;flex-direction:column;justify-content:center;align-items:flex-start}}
    .medcl-class-badge {{background:color-mix(in srgb,var(--accent) 10%,white);color:var(--accent);border-radius:7px;padding:6px 11px;font-size:12px;font-weight:650}}
    .medcl-class-info h2 {{font-size:30px;line-height:1.35;letter-spacing:-.7px;margin:20px 0 8px;padding:0;color:#203349}}
    .medcl-class-info .medcl-class-description {{font-size:14px;line-height:1.8;color:#64758b;margin:0 0 25px}}
    .medcl-class-category {{border-left:3px solid var(--accent);padding:3px 0 3px 16px;margin:0 0 26px}}
    .medcl-class-category small {{display:block;color:#64758b;font-size:12px;margin-bottom:7px}}
    .medcl-class-category strong {{font-size:24px;line-height:1.4;font-weight:650;color:#203349}}
    .medcl-class-meta {{display:grid;grid-template-columns:1fr 1fr;gap:18px 30px;border-top:1px solid #e6ecf2;padding-top:23px;width:100%;margin:0}}
    .medcl-class-meta dt {{font-size:12px;color:#64758b;margin-bottom:5px}}
    .medcl-class-meta dd {{font-size:14px;font-weight:550;margin:0;color:#304963}}
    .medcl-class-foot {{margin:20px 0 0;font-size:12px;color:#64758b;line-height:1.7}}
    @media(max-width:700px) {{.medcl-class-card {{grid-template-columns:1fr}} .medcl-class-info {{padding:25px}} .medcl-class-visual {{padding:20px}} .medcl-class-info h2 {{font-size:26px}}}}
    </style>
    <article class="medcl-class-card" style="--accent:{info['accent']}" aria-label="{name} 分类示例">
      <div class="medcl-class-visual">
        <div class="medcl-class-top"><span>{info['en']}</span><span class="medcl-class-rgb">RGB</span></div>
        <img class="medcl-class-image" src="data:image/png;base64,{pixels}" width="{width}" height="{height}" alt="{name} 示例图像：{category}">
        <div class="medcl-class-image-note"><span>示例图像</span><span>{width} × {height} px</span></div>
      </div>
      <div class="medcl-class-info">
        <span class="medcl-class-badge">{modality}</span>
        <h2>{name}</h2><p class="medcl-class-description">{escape(info['description'])}</p>
        <div class="medcl-class-category"><small>示例分类结果</small><strong>{category}</strong></div>
        <dl class="medcl-class-meta"><div><dt>图像分辨率</dt><dd>{width} × {height}</dd></div>
          <div><dt>类别编号</dt><dd>{label:02d}</dd></div><div><dt>数据集类别</dt><dd>{info['classes']} 类</dd></div><div><dt>图像通道</dt><dd>RGB · 彩色</dd></div></dl>
        <p class="medcl-class-foot">{'沿用素材中的类别编号。' if dataset == 'skin' else '切换上方数据集，浏览不同类型的医学影像。'}</p>
      </div>
    </article>'''
