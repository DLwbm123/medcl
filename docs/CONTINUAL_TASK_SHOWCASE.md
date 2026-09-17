# 持续学习逐任务示例

## 第二组模型示例接入（2026-09-17）

47 个任务均增加「示例 1 / 示例 2」切换，首页入口、分类卡片、切片与三维布局沿用原实现。示例 1 保留现有素材；示例 2 读取对应 independent 模型的预先导出结果，页面不展示训练方式名称。完整七类心脏仍是示例 1 的附加视图。

示例 2 的私有 NPZ 与原文件放在同一个 `$MEDCL_STATE_DIR/showcase/` 目录，在原逐任务文件名末尾加 `-independent`，例如 `classification-pathmnist-T1-independent.npz`、`segmentation-domain-T1-independent.npz`、`segmentation-weak-class-T2-independent.npz`、`registration-task-T4-independent.npz`。

复用已有数组结构：分类 `class_id` 填模型预测的全局类别；分割 `labels` 填模型预测标签，弱监督另外保留对应原始涂鸦；配准 `registered` 必须是真实配准输出。图像、预测与 spacing 必须对应同一病例和预览网格，不能用复制真值、固定影像或工程基线替代模型输出。权重路径、训练任务、病例索引、预处理与导出来源保存在私有来源记录中，不进入网页或 Git。

已从 my-gpu 与 jiangsuiyang 的既有训练记录核对权重，并通过 `scripts/export_showcase_predictions.py` 实际推理导出 **45 / 47** 组结果，安装到本机和服务器的私有展示目录。没有启动新训练、计算展示分数或创建评测作业。

| 类型 | 已接入示例 2 | 尚缺来源 |
|---|---|---|
| 分类 | PathMNIST 4、Skin 3、HyperKvasir 10 | 无 |
| 全监督分割 | Domain 6、Class 3、Task 2 | Task T3 肝脏、T4 脑肿瘤 |
| 弱监督分割 | Domain 6、Class 3、Task 4 | 无 |
| 配准 | OASIS、CTCT、NLST、MRCT 各 1 | 无 |

来源边界：分类 T2 以后使用逐任务重新初始化的 reference，T1 使用只训练过首任务的 FedAvg 检查点；全监督 Domain/Class 使用已完成的单任务模型，Task T1 使用全监督首阶段，Task T2 复用同一 UCL 二分类数据的 Domain 模型。弱监督使用对应单任务模型或没有先前任务的首阶段；Class T1 最终采用已完成 80 epoch 的单任务模型，未安装检索时发现的未完成候选。弱监督 Domain 的既有平台演示权重按原测试集选择，因此这些展示不能解释为留出集性能。配准使用单任务 reference，OASIS 使用仅完成首任务的普通顺序训练边界。后续持续学习阶段、其他监督类型和标签参考均未替代缺失的两项权重。

导出使用各训练环境的原始模型实现并严格加载参数。分类复用原输入缩放与任务类别掩码；分割推理前核对首个病例与原展示图像的对应关系，保留已记录的 spacing；配准固定取原 provider 的第一个测试图像对，导出真实 warped moving，按 provider 的轴排列转换为 ZYX 并保留 NIfTI spacing，不使用标签或关键点作为推理输入。来源记录为 `independent-*-provenance.private.json`，包括具体权重、任务、病例与运行路径，保留在私有目录。

接入检查：现有图库 AppTest 覆盖 47 × 2 个位置，使用刻意不同的测试数组检查实际分类卡片和三维 envelope；另检查无评测作业写入、缺失素材不回退、旧心脏入口与间距交互。`python -m unittest discover -s tests -p test_showcase.py -v` 的 7 项测试通过。测试数组仅存在于临时测试目录。45 个实际模型输出均通过素材结构检查，体数据通过 envelope 打包。本机浏览器抽查了实际分类卡片、分割切片及三维显示、配准的固定/移动/结果与棋盘格对比。未逐一人工查看全部 45 组预测。

服务器沿用原发布目录，仅更新 `medcl/showcase.py`；旧文件保留备份，只重新加载网页进程。服务器环境完成四类素材加载检查，网页进程为 RUNNING，公开 HTTPS 健康检查返回 `ok`。原始素材和完整七类心脏示例保留；选择尚缺的两项示例 2 时显示原有素材不可用提示。医学数组、权重与病例截图不进入公开 Git。

## 原逐任务素材

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
