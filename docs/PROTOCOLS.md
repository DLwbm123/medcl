# 评测协议与提交契约

## ID、顺序与阶段

任务 ID 是测试集身份；阶段是用户所选顺序中的 1-based 位置。例如 `[T3,T1,T2]` 的阶段 1 只见 T3，不是 T1。阶段输入必须覆盖协议要求的全部任务及样本。配置入队后由 SQLite 触发器禁止变更；未上传的阶段保持整行空值。测试资产版本及大小/修改时间在提交时记录，并在每个任务读取前复查；这是轻量变更检测，不是内容摘要或不可变快照。

域增量表示输入域变化；类别增量表示类别集合扩展；任务增量表示器官/模态或目标变化。该定义不自动决定输出头。共享输出头和任务指定输出头分开记录，后者仅接受预测，并明确已知任务 ID；都使用全局标签编码，评分器不依据真值重映射预测。

一级入口固定为分割、分类、配准；二级场景由管理员协议声明。分割可登记域/类/任务增量并记录全监督或弱监督训练条件，分类当前登记类别增量和逻辑客户端联邦评分，配准当前登记任务增量。协议用 `allowed_output_heads` 明确允许的 `shared` / `task-specific`；网页和提交后端使用同一 allow-list，不能靠修改请求绕过。类别增量必须使用共享全局标签空间；任务指定头只在管理员已登记且任务语义明确时开放。

分割的“全监督 / 弱监督”是提交者对外部训练条件的声明，随冻结配置保存并进入比较相容性检查。平台不读取训练集或训练日志，两类模型都使用相同冻结、完整标注的测试集评分。结果来源另以受限枚举保存；`untrained_baseline`、`trained_model_declared` 和 `external_predictions_unknown` 都不等于平台验证了训练过程，`synthetic` 则由通过 schema 校验的全合成协议决定。

类别增量默认只允许当前阶段已见类别输出，分割包含背景 0。任务 ID 不用于替用户校正预测。模型必须暴露注册协议的完整全局输出通道，推理再限制已见通道。未见任务开关仅在固定语义的共享输出协议中开放；无前向参照仍不计算 FWT。

## 数值定义

- 分类：worker 内部临时计算每个样本的正确指示量，再按样本数聚合为任务/逻辑客户端 Accuracy。提交者可见结果、SQLite `result` 和 JSON/HTML/CSV 均不保存或发布逐样本正误、样本/病例 ID、隐藏标签频数；任务宏平均、客户端宏平均与客户端样本加权准确率仍分别报告。
- 分割：先在每个病例所有切片/体素上计算各类 `(2*intersection + 1e-5)/(pred_count + target_count + 1e-5)`；同空为 1，单侧为空接近 0。主分数为病例内前景类宏平均，再对病例宏平均。背景 Dice 和含背景宏均值另列，不能混为主分数。HDF5 插值标签按参考 DenseDataset `astype(int64)` 截断，不做 0.5 阈值；模型基线对**输入图像**的 0.5 阈值是另一件事。
- 配准：对应点坐标差乘协议 spacing 后求欧氏距离，再对标志点、病例平均；TRE 单位 mm，越低越好。真实协议 v1 只接受已转入固定空间的有序 xyz 毫米坐标，spacing 必须为 `[1,1,1]`。未实现任意 voxel/world 变换，不猜坐标顺序或物理单位。
- 最终任务宏平均：最终阶段全部任务的等权平均，缺任一最终单元格即不可计算。
- BWT：`mean(R[T,j] - R[j,j])`，遍历前 T−1 个任务。TRE 反向为 `mean(R[j,j] - R[T,j])`，正值表示改善。
- 遗忘：高向指标使用 `mean(max(R[i,j], i=j..T−1) - R[T,j])`；低向指标使用 `mean(R[T,j] - min(R[i,j], i=j..T−1))`。要求相关历史完整，允许负值表示改善，不截断为 0。
- BWTR：仅对高向指标且历史对角线为正时提供相对 BWT；低向 TRE 不套用该比率。
- FWT / RMA：当前注册协议没有随机初始化或独立训练参照，保持不可计算，不能从最终结果倒推。

逻辑客户端先按匿名病例/图像序号固定轮转。全局评分使用全部有效病例/样本；客户端宏平均只包含有样本客户端。加权分类准确率 `sum(n_c*accuracy_c)/sum(n_c)`；分割/配准不伪装成这一分类指标。差异为总体标准差与最大最小差；低向指标的最差客户端取最大值。没有测试样本的客户端分数为 null、样本数为 0；未评测单元格的样本数未知，导出留空。

## 模型与数组

| 已审核结构 | float32 safetensors 键 | 输入与输出 |
|---|---|---|
| `linear-classifier-v1` | `weight[C,D]`, `bias[C]` | `N×...` 按 C 顺序展开；uint8 除以 255，其余不额外标准化；输出 `N` 个全局整数标签 |
| `pixel-linear-v1` | `weight[C,1]`, `bias[C]` | 单通道 `N×H×W`，逐像素线性 logits；输出同形标签 |
| `point-translation-v1` | `offset[D]` | `N×K×D` 的固定空间 moving points；输出 points+offset；真实协议 D=3 |

C 必须等于注册全局最大标签+1。参数完全从上传 safetensors 读取；平台不训练、估计或改写参数。示例分类器/阈值器/零位移从未训练。UNet、EfficientNet、SAMCL、任意自定义依赖等均不在模型执行名单内，可先在用户自己的可信环境生成预测。

预测 JSON 使用 `medcl.predictions.v1`；NPZ/ZIP 每个 task 为 `task__ids.npy`（一维字符串）和 `task__pred.npy`（纯数值）。分类形状 `[N]`；分割 `[N,H,W]`；标志点 `[N,K,D]`。每个样本 ID 一次，顺序可不同；平台对齐后严格检查形状/数值/输出类别。不接受概率图代替整数标签，不自动 argmax 或阈值化预测。`registration-volume` 只接受预测包，并可额外含 `task__registered.npy` 与 `task__warped_prediction.npy`，均为 `[N,Z,Y,X]` 且与 fixed display grid 完全同形；registered 是有限 scalar volume，warped prediction 是 uint16 范围内的整数预测 labelmap。它们只供定性查看，不参与 TRE。

HDF5 管理员输入键为 `test_images`、`test_labels`（HWN）和 `patient_info_test`（每病例最后一张切片的 inclusive index）；边界必须是一维、非空、有限整数、首项非负、严格递增且末项覆盖最后一张切片，生成的每个半开区间都必须非空。标签读取后先按参考实现截断为整数，再仅对非背景应用可选 `label_shift`；实际标签必须属于 `{0} ∪ classes`，防止协议遗漏前景类。`voxel_spacing_zyx` 缺失时只按 `[1,1,1]` 索引空间展示，不声称物理尺寸或病人方向。分类 NPY/MedMNIST 仅用 test split；可按全局类分任务，匿名样本 ID 保留原 test 数组行号，但不会回写结果。

标志点配准 NPZ 使用且仅使用 `moving_points`、`fixed_points`。体数据配准管理员 NPZ 使用且仅使用 `fixed_volumes`、`moving_volumes`、`moving_points`、`fixed_points`，体数据为同形有限 `[N,Z,Y,X]`，点为 `[N,K,3]`。fixed points 只交给评分器；主指标始终为 TRE。registered 图像和 warped prediction 由提交者在平台外生成并重采样到 fixed display grid 后上传；平台不执行配准、插值、重采样或形变场推理。

分割预览保存归一化 image volume 与 prediction labelmap；页面给出轴向/冠状/矢状类切面、预测叠加、Cornerstone Labelmap 三维显示和对应病例 Dice。体数据配准预览保存 fixed、moving、可选 registered、可选 warped prediction，以及 moving/predicted points；页面给出切面、融合与预测三维体，TRE 仍来自隐藏 fixed points。没有 registered 时明确退化为 Fixed + Moving，不伪造配准结果。

每个阶段任务最多保存 3 个私有病例预览。单轴不超过 128，单体素数不超过 1,000,000，按确定性步长下采样；scalar volume 用该预览体的 1/99 百分位归一化为 uint8，预测 labelmap 保留整数标签。spacing 随下采样步长更新。若协议未提供 spacing，则使用 `[1,1,1]` 并标记 `index-space-default`；若未同时提供 origin/direction，则查看器标记 index-space，三视图标题仅为 axial-like / coronal-like / sagittal-like，病人方向未经验证。

用于可视化的原始/fixed/moving/registered 影像和预测 labelmap 会进入本机浏览器内存。隐藏分割真值、fixed points、fixed truth segmentation 均不进入浏览器信封、预览 NPZ、完整下载报告或公开聚合。结果报告只含数值、元数据和私有预览引用，不嵌入体数据。预览读取限制路径、文件/展开大小、键、dtype、shape、有限值和 schema 版本；损坏预览只降级该图，不改变已保存数值评分。Cornerstone3D 查看器是论文范围内的只读单病例展示，不是完整 ITK-SNAP 替代品。

## 管理员协议 schema

配置加载时集中校验 1–12 个唯一安全 ID 任务、严格布尔值、任务类型固定的 metric/direction/unit、资产格式、输出头 allow-list 及必填绝对路径。分类类别增量任务必须使用一致的 `all_classes`、互斥且完整覆盖的 `classes` 和完整 `class_names`；分割必须显式登记背景 0、非空前景类、可选非负 `label_shift` 及可选正数 `voxel_spacing_zyx`；配准不得携带类别语义。标志点 v1 只接受 `fixed-space xyz, mm` 与 `[1,1,1]`；体数据 v1 只接受 `fixed-display-grid xyz, mm`，其点坐标已经是毫米，`spacing` 必须省略或为 `[1,1,1]`。任一合成任务要求整个协议 `synthetic=true` 且全部任务均为合成格式；专供浏览器验收的非对称 `registration-volume` 测试资产可作为全合成协议登记，但必须由管理员明确标记，不能混入真实协议。待接入协议可为空，但永远不进入 ready 状态。

本论文版本不恢复 DICOM patient orientation；`direction_xyz` 仅接受 identity matrix（允许 `1e-6` 以内的浮点误差），其他 affine 直接拒绝。

## 比较与可追溯性

比较校验 evaluator_version、基准版本/协议、真实或合成属性、任务顺序、测试资产轻量元数据、客户端版本/数量、分割监督声明、未见任务开关、输出头与指标条件。该相容性不能识别“同大小且恢复原 mtime”的内容替换；发生资产异常时需管理员确认并新建评测。方法名称、模型还是预测的输入方式不改变测试条件；原始来源仍随本地报告保存。无最终阶段的对比保留空值，不以其他阶段替代。

预测模式无法从预测数组独立验证模型来源；结构化 provenance 只记录受限类别和是否由平台验证，不要求用户上传训练日志来伪证明声明。模型模式的同一阶段权重会用于所有任务/客户端，评分只汇总数值、不聚合参数。公开 aggregate v2 不复制完整 config，而是白名单重建 run/benchmark 标识、来源类别、全局 cells/matrix/持续学习指标和固定安全警示；方法自由文本、管理员 description/source、病例/客户端明细和资产元数据均排除。

前端生产包随 Python 包预构建，不从 CDN 加载脚本，也不要求生产机安装 Node.js。修改 TypeScript/CSS 后必须重新执行 typecheck、Vitest 和 Vite build，并提交新的 `medcl_cornerstone/frontend/build`；只改源码而不重建不会影响生产页面。
