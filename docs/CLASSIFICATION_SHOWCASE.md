# 分类示例图像库

当前入口已扩展为每个持续学习任务各一个示例，见 [逐任务示例](CONTINUAL_TASK_SHOWCASE.md)。下文保留首次三数据集图像库的历史验收记录。

2026-09-07；起始本地与远端提交 `5f166dc10b53004f8ab10ee4d871c274d72edf2f`。
本记录更新分类示例的当前行为，不改写历史验收。

入口：主页 → 分类示例。可直接切换 PathMNIST、Skin Cancer、HyperKvasir，每套展示一个训练集首图及其原始类别。使用统一的左右图像/信息卡片，提供影像类型、突出显示的类别、原生网格尺寸及数据集类别数；700 px 以下自动上下排列。HTML/CSS 仅作用于分类卡片，切换仍使用原生 Streamlit 控件。

| 示例 | 准备好的源图像网格 | 类别范围 | 当前首图类别 |
|---|---|---|---|
| PathMNIST | 128×128 RGB | 0–8 | 0，脂肪组织 |
| Skin Cancer | 128×128 RGB | 0–5 | 1，保留数字编号 |
| HyperKvasir20 | 224×224 RGB | 0–19 | 7，retroflex-stomach / 胃内反转视图 |

PathMNIST 直接读取既有 `pathmnist_128_npy`，未由 28×28 放大生成。三套示例均不对素材数组做 resize、增强或重标；浏览器按布局尺寸显示。Skin Cancer 的准备好数组没有可核实的六类病名映射，不能借其他皮肤数据集的映射猜测名称。HyperKvasir 编号已核对对应的 `class_mapping.json`。

## 管理员准备

```sh
python scripts/prepare_showcase.py --classification-gallery \
  --classification /private/path/pathmnist_128_npy \
  --skin /private/path/skin_data_all_new \
  --hyperkvasir /private/path/hyperkvasir20_224_npy > /private/path/classification-gallery.tar
```

私有归档包含 `classification.npz`、`classification-skin.npz`、`classification-hyperkvasir.npz`、`classification-provenance.private.json`；解包到 `$MEDCL_STATE_DIR/showcase/`（默认 `~/.local/state/medcl/showcase/`）。归档内文件权限 0600。当前管理员已在本地安装素材，旧 28×28 示例另行私有备份。原始医学图像、来源绝对路径与实际病例截图不进入公开 Git。

此图像库为参考展示，不创建推理、评测作业、置信度或评分。只扩展管理员素材加载路径，不修改评测的传输规则。分割和配准组件未改动。

## 验证

- `/opt/miniconda3/bin/python3 -m unittest discover -s tests -v`：58 项通过。新增/扩展检查覆盖原生分辨率要求、图像/标签不变、数据集对应的形状与编号边界、三套界面切换及取消选择后的有效默认值、HTML 中实际嵌入的 PNG 像素、展示不创建评分作业。
- 真实 Streamlit 8501 中浏览三套图像，检查当前类别、分辨率和样式；浏览器报告 PathMNIST 的 `naturalWidth/naturalHeight` 均为 128。桌面 1280 宽与窄屏 640 宽均检查布局，临时尺寸覆盖已恢复。
- 实际截图保存在管理员本地 `.local/classification-evidence/`，未公开医学图像。本轮只修改 Python/静态样式，不涉及 TypeScript，因此无需重建未变更的 Cornerstone bundle，也未将上轮前端测试计为本轮测试。
