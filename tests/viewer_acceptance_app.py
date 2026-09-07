"""Run with streamlit; synthetic-only acceptance using the production component."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import streamlit as st
from medcl_cornerstone import pack_envelope, render

st.set_page_config(layout="wide")
st.title("MedCL · 预测三维验收（合成数据）")
left, middle, right = st.columns(3)
case = left.selectbox("测试病例", ["两个对象 · 2 / 7", "七前景 · 1–7", "uint16 · 2 / 513", "空预测", "单层薄体", "全体积前景", "配准"])
intensity = middle.selectbox("原图强度", ["原始合成体", "全零", "高对比体"])
if right.button("重复挂载"):
    st.session_state["mount_revision"] = st.session_state.get("mount_revision", 0) + 1
shape = (1, 40, 48) if case == "单层薄体" else (32, 40, 48)
z, y, x = np.indices(shape)
image = np.clip(20 + x * 3 + y + z * 2, 0, 255).astype(np.uint8)
if intensity == "全零": image[:] = 0
elif intensity == "高对比体": image = np.where((x // 4 + y // 4 + z // 4) % 2, 255, 0).astype(np.uint8)
prediction = np.zeros(shape, dtype=np.uint16)
if case == "七前景 · 1–7":
    for label, (cx, cy, cz) in enumerate([(10,10,9),(24,10,12),(38,10,9),(10,28,21),(24,28,18),(38,28,21),(24,19,26)], 1):
        prediction[((x-cx)/4)**2 + ((y-cy)/5)**2 + ((z-cz)/4)**2 <= 1] = label
elif case == "单层薄体":
    prediction[:, 8:19, 7:18] = 2
    prediction[:, 22:32, 28:40] = 7
elif case == "全体积前景": prediction[:] = 7
elif case != "空预测":
    prediction[((x-13)/5)**2 + ((y-14)/8)**2 + ((z-11)/6)**2 <= 1] = 2
    prediction[18:27, 23:31, 29:39] = 513 if case.startswith("uint16") else 7
if case == "配准":
    volumes = [("fixed", "scalar", image), ("moving", "scalar", np.roll(image, 4, axis=2)),
               ("registered", "scalar", image.copy())]
else:
    if not case.startswith("uint16"): prediction = prediction.astype(np.uint8)
    volumes = [("image", "scalar", image), ("prediction", "labelmap", prediction)]
envelope = pack_envelope(viewer_mode="registration" if case == "配准" else "segmentation",
    volumes=volumes, spacing_zyx=[2.5, 1.25, 0.75], spacing_source="protocol", origin_xyz=[11, -7, 3],
    segments=[int(v) for v in np.unique(prediction) if v], context={"case_id": "synthetic-acceptance"})
state = render(envelope, key=f"synthetic-acceptance-viewer-{st.session_state.get('mount_revision', 0)}")
st.write(state)

# Synthetic-only, lossless framebuffer evidence. This control is not in the product.
st.html('''
<button id="export-viewer-pixels">导出验收画布 PNG</button>
<textarea id="viewer-pixel-evidence" aria-label="无损画布证据" rows="2" style="width:100%"></textarea>
<script>
document.getElementById('export-viewer-pixels').onclick = () => {
  const roots = [document], images = {};
  for (let i = 0; i < roots.length; i++) {
    for (const pane of roots[i].querySelectorAll('[aria-label]')) {
      const name = pane.getAttribute('aria-label');
      if (!['预测分割 · 3D', 'Z / axial-like'].includes(name)) continue;
      const canvas = pane.querySelector('canvas');
      if (canvas) images[name] = canvas.toDataURL('image/png');
    }
    for (const el of roots[i].querySelectorAll('*')) if (el.shadowRoot) roots.push(el.shadowRoot);
  }
  document.getElementById('viewer-pixel-evidence').value = JSON.stringify(images);
};
</script>
''', unsafe_allow_javascript=True)
