# 持续学习逐任务示例

2026-09-07；本轮起始本地与远端 SHA：`3aaf8e81448ac5b51baa5e9e1847bae408ddd355`。

主页的分类、全监督、弱监督和配准示例均可选择「持续学习任务」。分割另有「持续学习场景」；从任务中心打开时沿用选中的场景。每个任务使用独立素材，缺失时提示不可用，不回退到其他任务病例。

| 类型 | 场景或数据集 | 任务与示例数 |
|---|---|---|
| 分类 | PathMNIST / Skin Cancer / HyperKvasir20 | 4 / 3 / 10 |
| 全监督分割 | Domain-CL / Class-CL / Task-CL | 6 / 3 / 4 |
| 弱监督分割 | Domain-CL / Class-CL / Task-CL | 6 / 3 / 4 |
| 配准 | OASIS / CTCT / NLST / MRCT | 各 1，共 4 |

共 47 个任务示例。Domain-CL 顺序为 BIDMC、HK、ISBI、UCL、ISBI-1.5、I2CVB；Class-CL 三组前景为 1/2/3、4/5、6/7；Task-CL 为左心房、前列腺、肝脏、脑肿瘤。分类按现有类别任务划分选择该组首个训练图像；PathMNIST 与 Skin 保留原生 128×128，HyperKvasir 保留 224×224。分类卡片展示该图像的类别。

按用户确认，本轮继续使用标签参考展示，不运行模型、不产生置信度、评测作业或分数。配准的「对齐参考」仍为固定影像的目标显示状态，不能视为计算得到的变形结果。公开代码中保留这一来源说明。

## 分割与几何

每个 H5 取首个训练病例，沿用已有协议的整数解码和前景编号偏移。类增量 T2/T3 的局部 1/2 分别映射为全局 4/5、6/7；原文件不修改。已有稀疏涂鸦已经使用全局编号，不重复偏移；准备时检查所有已标注体素与同一病例的完整标签一致。未知值 -100 在私有展示 NPZ 中编码为 255，背景 0 单独显示，避免把真实前景 4 当成未知。

保留 ZYX 转换、预览网格和步长 spacing。H5 无真实物理信息，使用索引空间提示；配准使用已准备的共同网格及 NIfTI spacing，仍不宣称患者方向。预览按最近邻步长采样，不平滑、补全或删除对象。

3D 复用现有逐标签 Surface 查看器，MPR 使用 image 与 prediction，分割第四视口只显示 prediction 派生表面。任务/场景/监督方式具有独立的组件 key 与病例 ID，切换后重新挂载；没有修改前端渲染实现或评测传输规则。

完整七类心脏保留为附加示例：全监督示例 → Class-CL → T3 →「查看最终七类完整心脏示例」。逐任务默认展示该任务自己的标签组。

## 管理员准备

在仓库根目录、已有 numpy/h5py/nibabel 环境运行：

```sh
PYTHONPATH=. python scripts/prepare_showcase.py --task-gallery \
  --classification /private/path/pathmnist_128_npy \
  --skin /private/path/skin_data_all_new \
  --hyperkvasir /private/path/hyperkvasir20_224_npy \
  --assets /private/path/MedCL/assets \
  --segmentation-root /private/path/CL_Benchmark/data \
  --sparse-root /private/path/sparse_annotations > /private/path/tasks.tar
```

归档包含 47 个 NPZ 与 `tasks-provenance.private.json`，文件权限 0600。安装到 `$MEDCL_STATE_DIR/showcase/`，默认 `~/.local/state/medcl/showcase/`。原有完整心脏素材另由 `--cardiac` 准备。当前素材从 my-gpu 只读准备并安装到本机；医学数组、来源绝对路径与真实病例截图不进入公开 Git。

## 本轮验证与限制

- 根目录 `python3 -m unittest discover -s tests -v` 使用 `/opt/miniconda3/bin/python3`，59 项通过。逐任务 AppTest 遍历全部 47 项，检查图像/数组与任务对应、分类实际嵌入 PNG 像素、envelope 标签和 task_id、缺失任务不回退、稀疏未知值及场景入口。
- 实际重启 Streamlit 8501。浏览器检查 Domain T1→T6 三维病例切换并旋转 T6；Class T2 的 4/5 表面、T3 的 6/7 及单类 7 筛选；弱监督 Class T2 涂鸦；HyperKvasir T10；MRCT T4 的固定/参考融合与独立 moving 体绘制。
- 运行截图仅保存在管理员本机 `.local/task-showcase-evidence/`，不公开病例影像。全部 47 个已安装 NPZ 均成功加载。
- 本轮未修改 TypeScript、依赖或生产 bundle，未把历史 npm 验证、image-independence 像素比较记为本轮重跑；未逐一人工旋转全部 30 个分割/配准示例，也未运行模型推理。自动逐任务覆盖与上述实际浏览器抽查分别报告。
