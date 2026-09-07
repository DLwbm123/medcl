"""Curated reference replays, kept separate from scoring and frozen test previews.

Only the administrator-prepared, owner-only showcase files are read here. Source
labels are intentionally used for this gallery, never as model predictions in jobs.
"""

from functools import lru_cache
import zipfile

import numpy as np

from medcl.benchmarks import state_path
from medcl.classification_showcase import DATASETS, card_html
from medcl_cornerstone import pack_envelope, render as render_volume

EXAMPLES = {
    "classification": {"kind": "classification", "title": "医学影像分类",
                       "description": "病理、皮肤与消化道内镜图像的分类展示。"},
    "segmentation-full": {"kind": "segmentation", "title": "全监督分割",
                          "description": "原始影像、区域叠加与三维结构浏览。"},
    "segmentation-cardiac": {"kind": "segmentation", "title": "心脏七结构分割",
                             "description": "类增量最终七类的切面与三维结构展示。"},
    "segmentation-weak": {"kind": "segmentation", "title": "弱监督分割",
                          "description": "稀疏涂鸦、完整分割区域与三维结构浏览。"},
    "registration": {"kind": "registration", "title": "医学影像配准",
                     "description": "固定影像、移动影像、对齐参考与融合对比。"},
}
COLORS = np.array([[38, 200, 122], [255, 181, 71], [96, 165, 250], [207, 122, 232],
                   [255, 112, 137], [67, 217, 214], [232, 222, 85]], dtype=np.uint8)


def load_example(example, *, dataset="pathmnist"):
    if example not in EXAMPLES:
        raise ValueError("Unknown showcase")
    if dataset not in DATASETS:
        raise ValueError("Unknown classification dataset")
    filename = DATASETS[dataset]["file"] if example == "classification" else example
    path = state_path() / "showcase" / f"{filename}.npz"
    arrays = _load(str(path), path.stat().st_mtime_ns)
    if ("class_id" in arrays) != (example == "classification") or (
            "fixed" in arrays) != (example == "registration") or (
            "scribble" in arrays) != (example == "segmentation-weak"):
        raise ValueError("Showcase task mismatch")
    if example == "classification":
        info = DATASETS[dataset]
        if arrays["image"].shape != (info["size"], info["size"], 3) or not 0 <= int(arrays["class_id"]) < info["classes"]:
            raise ValueError("Classification dataset mismatch")
    return arrays


@lru_cache(maxsize=7)
def _load(path, mtime):
    with zipfile.ZipFile(path) as archive:
        if sum(item.file_size for item in archive.infolist()) > 32 * 1024 * 1024:
            raise ValueError("Showcase too large")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if "class_id" in arrays:
        if (set(arrays) != {"image", "class_id"} or arrays["image"].shape not in ((128, 128, 3), (224, 224, 3))
                or arrays["image"].dtype != np.uint8 or arrays["class_id"].shape != ()
                or arrays["class_id"].dtype.kind not in "iu" or not 0 <= int(arrays["class_id"]) < 20):
            raise ValueError("Invalid classification example")
    else:
        keys = set(arrays) - {"spacing_source"}
        if str(arrays.get("spacing_source", "index-space-default")) not in ("protocol", "index-space-default"):
            raise ValueError("Invalid spacing source")
        expected = {"image", "labels", "spacing"} if "image" in keys else {"fixed", "moving", "registered", "spacing"}
        if keys not in (expected, expected | {"scribble"}):
            raise ValueError("Invalid showcase fields")
        volumes = [a for k, a in arrays.items() if k not in ("spacing", "spacing_source")]
        shape = volumes[0].shape
        if (len(shape) != 3 or min(shape) <= 0 or max(shape) > 160
                or any(a.shape != shape or a.dtype != np.uint8 for a in volumes)
                or arrays["spacing"].shape != (3,) or not np.isfinite(arrays["spacing"]).all()
                or np.any(arrays["spacing"] <= 0)):
            raise ValueError("Invalid showcase volume")
        if "labels" in arrays and not np.isin(arrays["labels"], range(8)).all():
            raise ValueError("Invalid labels")
        if "scribble" in arrays and not np.isin(arrays["scribble"], [0, 1, 2, 3, 4]).all():
            raise ValueError("Invalid scribble")
    return arrays


def overlay(image, labels, opacity=0.5, *, scribble=False):
    result = np.repeat(image[..., None], 3, axis=-1)
    for label, color in enumerate(COLORS[:3] if scribble else COLORS, 1):
        mask = labels == label
        result[mask] = ((1 - opacity) * result[mask] + opacity * color).astype(np.uint8)
    if scribble:
        result[labels == 0] = [220, 226, 235]
    return result


def checkerboard(fixed, other, tile=12):
    y, x = np.indices(fixed.shape)
    return np.where((y // tile + x // tile) % 2 == 0, fixed, other)


def volume_envelope(example, arrays):
    segmentation = EXAMPLES[example]["kind"] == "segmentation"
    volumes = [("image", "scalar", arrays["image"]), ("prediction", "labelmap", arrays["labels"])] if segmentation else [
        (name, "scalar", arrays[name]) for name in ("fixed", "moving", "registered")]
    return pack_envelope(viewer_mode="segmentation" if segmentation else "registration",
                         volumes=[(name, role, np.ascontiguousarray(a)) for name, role, a in volumes],
                         spacing_zyx=arrays["spacing"], spacing_source=str(arrays.get("spacing_source", "index-space-default")),
                         segments=[int(x) for x in np.unique(arrays["labels"]) if x > 0] if segmentation else [],
                         context={"case_id": f"example-{example}", "downsampled": True})


def render(st, example):
    if example in ("segmentation-full", "segmentation-cardiac"):
        example = st.selectbox("示例病例", ["segmentation-full", "segmentation-cardiac"],
                               index=int(example == "segmentation-cardiac"),
                               format_func=lambda value: "前列腺 · 单前景" if value.endswith("full") else "心脏 · 七前景",
                               key="showcase-full-case")
    definition = EXAMPLES[example]
    st.title(f"{definition['title']} · 示例展示")
    st.caption("从组织纹理到皮肤病变与内镜视野，浏览三种医学影像的分类示例。" if example == "classification" else "预置病例展示 · 可切换视图与浏览细节")
    dataset = "pathmnist"
    if example == "classification":
        if st.session_state.get("showcase-classification-dataset") not in DATASETS:
            st.session_state["showcase-classification-dataset"] = "pathmnist"
        dataset = st.segmented_control("影像类型", list(DATASETS),
            format_func=lambda key: f"{DATASETS[key]['name']} · {DATASETS[key]['modality']}",
            key="showcase-classification-dataset")
    if example == "registration":
        st.caption("使用固定与移动脑 MRI 展示对齐关系，参考视图呈现目标对齐状态。")
    try:
        arrays = load_example(example, dataset=dataset)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        st.info("此示例素材暂不可用，请联系管理员准备展示病例。")
        return
    if example == "classification":
        st.html(card_html(dataset, arrays))
        return

    if example == "segmentation-cardiac":
        st.caption("类增量最终七类 · 保留原始标签编号 1–7 · 0 为背景")

    view = st.radio("展示视图", ["切片对比", "三维浏览"], horizontal=True, key=f"showcase-view-{example}")
    if view == "三维浏览":
        try:
            state = render_volume(volume_envelope(example, arrays), key=f"showcase-volume-{example}")
            if not isinstance(state, dict) or not state.get("viewer_error_code"):
                st.caption("滚轮浏览切片；拖动旋转三维视图，可调节叠加透明度。")
                return
        except (OSError, RuntimeError, TypeError, ValueError):
            pass
        st.info("三维视图暂不可用，已显示切片对比。")

    segmentation = definition["kind"] == "segmentation"
    image = arrays["image" if segmentation else "fixed"]
    controls = st.columns([1, 2, 1] if segmentation else [1, 2], gap="medium")
    axis = controls[0].selectbox("切面", [0, 1, 2], format_func=lambda i: ["Z 切面", "Y 切面", "X 切面"][i],
                        key=f"showcase-axis-{example}")
    default = image.shape[axis] // 2
    if segmentation:
        default = int(np.argmax(np.count_nonzero(arrays["labels"], axis=tuple(i for i in range(3) if i != axis))))
    index = controls[1].slider("切片位置", 0, image.shape[axis] - 1, default, key=f"showcase-slice-{example}-{axis}")
    cut = lambda name: np.take(arrays[name], index, axis=axis)
    if segmentation:
        opacity = controls[2].slider("区域透明度", 0.0, 1.0, 0.5, 0.05, key=f"showcase-opacity-{example}")
        panels = [("原始影像", cut("image"))]
        if "scribble" in arrays:
            panels.append(("稀疏涂鸦", overlay(cut("image"), cut("scribble"), 1, scribble=True)))
        panels.append(("分割区域", overlay(cut("image"), cut("labels"), opacity)))
        st.caption("绿色：分割区域" if example.endswith("full") else
                   "绿色：标签 1 · 橙色：标签 2 · 蓝色：标签 3 · 紫色：标签 4 · 粉色：标签 5 · 青色：标签 6 · 黄色：标签 7"
                   if example == "segmentation-cardiac" else "绿色：右心室 · 橙色：心肌 · 蓝色：左心室 · 浅灰：背景涂鸦")
    else:
        panels = [("固定影像", cut("fixed")), ("移动影像", cut("moving")), ("对齐参考", cut("registered"))]
    for column, (label, pixels) in zip(st.columns(len(panels), gap="medium"), panels):
        column.image(pixels, caption=label, width="stretch")
    if not segmentation:
        st.subheader("对齐细节")
        mode = st.radio("对比方式", ["棋盘格", "彩色融合"], horizontal=True)
        for column, name, label in zip(st.columns(2), ("moving", "registered"), ("原始图像对", "参考对齐状态")):
            pixels = checkerboard(cut("fixed"), cut(name)) if mode == "棋盘格" else np.stack(
                [cut("fixed"), cut(name), cut(name)], axis=-1)
            column.image(pixels, caption=label, width="stretch")
