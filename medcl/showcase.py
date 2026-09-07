"""Curated reference replays, kept separate from scoring and frozen test previews.

Only the administrator-prepared, owner-only showcase files are read here. Source
labels are intentionally used for this gallery, never as model predictions in jobs.
"""

from functools import lru_cache
import zipfile

import numpy as np

from medcl.benchmarks import state_path
from medcl.classification_showcase import DATASETS, card_html
from medcl.showcase_tasks import SCENARIOS, task_specs
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


def load_example(example, *, dataset="pathmnist", scenario="domain", task_id=None):
    if example not in EXAMPLES:
        raise ValueError("Unknown showcase")
    if dataset not in DATASETS:
        raise ValueError("Unknown classification dataset")
    filename = DATASETS[dataset]["file"] if example == "classification" else example
    if task_id is not None:
        if scenario not in SCENARIOS:
            raise ValueError("Unknown showcase scenario")
        specs = task_specs(EXAMPLES[example]["kind"], scenario=scenario, dataset=dataset)
        spec = next((item for item in specs if item["id"] == task_id), None)
        if spec is None:
            raise ValueError("Unknown continual task")
        filename = spec["file"]
        if example == "segmentation-weak":
            filename = filename.replace("segmentation-", "segmentation-weak-", 1)
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
        if task_id is not None and int(arrays["class_id"]) not in spec["classes"]:
            raise ValueError("Classification task mismatch")
    return arrays


@lru_cache(maxsize=12)
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
        keys = set(arrays) - {"spacing_source", "scribble_ignore"}
        if str(arrays.get("spacing_source", "index-space-default")) not in ("protocol", "index-space-default"):
            raise ValueError("Invalid spacing source")
        expected = {"image", "labels", "spacing"} if "image" in keys else {"fixed", "moving", "registered", "spacing"}
        if keys not in (expected, expected | {"scribble"}):
            raise ValueError("Invalid showcase fields")
        volumes = [a for k, a in arrays.items() if k not in ("spacing", "spacing_source", "scribble_ignore")]
        shape = volumes[0].shape
        if (len(shape) != 3 or min(shape) <= 0 or max(shape) > 160
                or any(a.shape != shape or a.dtype != np.uint8 for a in volumes)
                or arrays["spacing"].shape != (3,) or not np.isfinite(arrays["spacing"]).all()
                or np.any(arrays["spacing"] <= 0)):
            raise ValueError("Invalid showcase volume")
        if "labels" in arrays and not np.isin(arrays["labels"], range(8)).all():
            raise ValueError("Invalid labels")
        ignore = arrays.get("scribble_ignore", np.asarray(4))
        if ignore.shape != () or ignore.dtype.kind not in "iu" or int(ignore) not in (4, 255):
            raise ValueError("Invalid scribble ignore value")
        if "scribble_ignore" in arrays and "scribble" not in arrays:
            raise ValueError("Scribble metadata without annotations")
        if "scribble" in arrays and not np.isin(arrays["scribble"], list(range(4 if int(ignore) == 4 else 8)) + [int(ignore)]).all():
            raise ValueError("Invalid scribble")
    return arrays


def overlay(image, labels, opacity=0.5, *, scribble=False, ignore=4):
    result = np.repeat(image[..., None], 3, axis=-1)
    for label, color in enumerate(COLORS[:3] if scribble and ignore == 4 else COLORS, 1):
        mask = labels == label
        result[mask] = ((1 - opacity) * result[mask] + opacity * color).astype(np.uint8)
    if scribble:
        result[labels == 0] = [220, 226, 235]
    return result


def checkerboard(fixed, other, tile=12):
    y, x = np.indices(fixed.shape)
    return np.where((y // tile + x // tile) % 2 == 0, fixed, other)


def volume_envelope(example, arrays, *, case_id=None, task_id=None):
    segmentation = EXAMPLES[example]["kind"] == "segmentation"
    volumes = [("image", "scalar", arrays["image"]), ("prediction", "labelmap", arrays["labels"])] if segmentation else [
        (name, "scalar", arrays[name]) for name in ("fixed", "moving", "registered")]
    return pack_envelope(viewer_mode="segmentation" if segmentation else "registration",
                         volumes=[(name, role, np.ascontiguousarray(a)) for name, role, a in volumes],
                         spacing_zyx=arrays["spacing"], spacing_source=str(arrays.get("spacing_source", "index-space-default")),
                         segments=[int(x) for x in np.unique(arrays["labels"]) if x > 0] if segmentation else [],
                         context={"case_id": case_id or f"example-{example}", "downsampled": True,
                                  **({"task_id": task_id} if task_id else {})})


def render(st, example):
    definition = EXAMPLES[example]
    st.title(f"{definition['title']} · 展示")
    st.caption("从组织纹理到皮肤病变与内镜视野，浏览三种医学影像的分类。" if example == "classification" else "预置病例展示 · 可切换视图与浏览细节")
    dataset = "pathmnist"
    if example == "classification":
        if st.session_state.get("showcase-classification-dataset") not in DATASETS:
            st.session_state["showcase-classification-dataset"] = "pathmnist"
        dataset = st.segmented_control("影像类型", list(DATASETS),
            format_func=lambda key: f"{DATASETS[key]['name']} · {DATASETS[key]['modality']}",
            key="showcase-classification-dataset")
    scenario = "domain"
    kind = definition["kind"]
    if kind == "segmentation":
        scenario = st.selectbox("持续学习场景", list(SCENARIOS), format_func=SCENARIOS.get,
                                key=f"showcase-scenario-{example}")
    specs = task_specs(kind, scenario=scenario, dataset=dataset)
    choices = {item["id"]: item for item in specs}
    task_id = st.selectbox("持续学习任务", list(choices),
        format_func=lambda value: f"{value} · {choices[value]['name']}",
        key=f"showcase-task-{example}-{scenario}-{dataset}")
    st.caption(f"共 {len(specs)} 个任务 · 每任务 1 组图像与结果 · 当前 {task_id}：{choices[task_id]['name']}")
    whole_heart = kind == "segmentation" and scenario == "class" and task_id == "T3" and st.checkbox(
        "查看最终七类完整心脏", key=f"showcase-whole-heart-{example}")
    if whole_heart:
        example = "segmentation-cardiac"
    view_key = f"{example}-{scenario}-{dataset}-{task_id}"
    try:
        arrays = load_example(example, dataset=dataset, scenario=scenario, task_id=None if whole_heart else task_id)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        st.info("此素材暂不可用，请联系管理员准备展示病例。")
        return
    if example == "classification":
        st.html(card_html(dataset, arrays))
        return

    if example == "segmentation-cardiac":
        st.caption("类增量最终七类 · 保留原始标签编号 1–7 · 0 为背景")

    view = st.radio("展示视图", ["切片对比", "三维浏览"], horizontal=True, key=f"showcase-view-{view_key}")
    if view == "三维浏览":
        try:
            state = render_volume(volume_envelope(example, arrays, case_id=f"example-{view_key}", task_id=task_id),
                                  key=f"showcase-volume-{view_key}")
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
                        key=f"showcase-axis-{view_key}")
    default = image.shape[axis] // 2
    if segmentation:
        default = int(np.argmax(np.count_nonzero(arrays["labels"], axis=tuple(i for i in range(3) if i != axis))))
    index = controls[1].slider("切片位置", 0, image.shape[axis] - 1, default, key=f"showcase-slice-{view_key}-{axis}")
    cut = lambda name: np.take(arrays[name], index, axis=axis)
    if segmentation:
        opacity = controls[2].slider("区域透明度", 0.0, 1.0, 0.5, 0.05, key=f"showcase-opacity-{view_key}")
        panels = [("原始影像", cut("image"))]
        if "scribble" in arrays:
            panels.append(("稀疏涂鸦", overlay(cut("image"), cut("scribble"), 1, scribble=True, ignore=int(arrays.get("scribble_ignore", 4)))))
        panels.append(("分割结果", overlay(cut("image"), cut("labels"), opacity)))
        st.caption(" · ".join(f"{color}：标签 {label}" for label, color in enumerate(
            ("绿色", "橙色", "蓝色", "紫色", "粉色", "青色", "黄色"), 1) if label in arrays["labels"]) +
            (" · 浅灰：背景涂鸦" if "scribble" in arrays else ""))
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
