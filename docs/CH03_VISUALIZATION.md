# 第三章持续分割 Benchmark

实现及验收日期：2026-09-10。增量实现基线：`84bc195632ad16d1e123ef0cc0bf70e4ecfe2c01`，分支 `codex/medcl-evaluation-platform`；没有回退代码。

进入“方法比较”后默认显示“第三章基准结果”，可切换回原有“平台评测结果”。历史页始终标记“论文历史结果 · 第三章 · 非平台重新评测”。历史数据不写入 jobs，不参与平台评分、首页统计或混合排名。

## 运行与模块

```sh
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

启动只需要已提交的 `medcl/ui_assets/ch03/results.json` 和四幅结果图，不需要论文 ZIP、医学资产、数据盘、GPU、worker、Poppler 或 LaTeX。浏览器页面沿用 Streamlit 和 Altair，不新增前端工程；样式限定在比较模块。

| 文件 | 职责 |
|---|---|
| `app.py` | 比较来源入口；旧 completed-jobs 限制只在平台分支生效 |
| `medcl/reference_results.py` | 只读 schema、筛选、指标解释、协议、CL 前沿和矩阵派生值 |
| `medcl/benchmark_charts.py` | 纯 Altair 图表 |
| `medcl/benchmark_compare_ui.py` | 六个子视图、选择状态、范围说明、下载 |
| `medcl/reference_exports.py` | CSV、严格 JSON、真实 SVG/PNG、嵌入图表的离线 HTML |
| `scripts/import_ch03_reference.py` | 显式离线导入并对照种子；不在启动时调用 |
| `tests/test_reference_results.py` | 主表全部单元、72 格矩阵及顺序 SD 校验 |
| `tests/test_benchmark_charts.py` | 图形语义及实际渲染、导出内容校验 |
| `tests/test_benchmark_ui.py` | 无资产/0、1、2 jobs、筛选和原比较回归 |

数据导入需要本地 Poppler；导入命令接受本轮作者 ZIP 和核对种子。文件名不能用于判断论文版本，导入器验证附件 SHA-256、五项来源白名单及重复成员，然后独立解析 LaTeX/PDF 并与种子核对。

```sh
python3 scripts/import_ch03_reference.py \
  /path/to/PhD_Thesis.zip \
  /path/to/ch03_reference_seed.json
```

本轮仅新增已核验 JSON、四个结果 PDF 及其 PNG。没有新增整本论文、字体、医学体数据、私有路径或第四至六章材料。原有论文 ZIP 未改动。

## 已实现图表

| 子视图 | 内容 |
|---|---|
| 总览 | KPI；完整 A-Dice 横向排名及 SD；方法数量环图（默认不含两类参照）；可切换合法主指标 |
| 能力与资源 | BWTR 零基线点区间；RMA=1 点区间；Domain E-FWT；BWTR–RMA 散点；A-Dice–MPE / DRR 资源图及当前 CL 均值 Pareto 前沿；Class A-Dice–WCD 哑铃 |
| 跨场景 | 方法×场景热图，一次一个 A-Dice / BWTR / RMA，共用稳定方法 ID，明确未报告 |
| 鲁棒性 | 八种方法、两项指标的任务顺序 SD；两幅顺序研究原图；回放容量原图及 PDF 下载 |
| SAM 扩展 | 共享色标的 SAM / SAM-LoRA 双 6×6 热图、零中心差值图；选择测试任务的六阶段观测曲线；原图入口 |
| 数据与来源 | 完整原文表、独立 mean/sd、来源位置及 SHA、指标定义、协议、缺失原因、46 行原文映射 |

类别使用稳定颜色，并配合文字与散点形状。默认展示全部已报告 CL 方法，支持类别、多选方法、参照、SD 开关及重置；切换场景清理不合法方法/指标。鲁棒性及 SAM 仅有 Domain 数据，其他场景显示数据边界及切换入口。

## 数值语义

主表存储 `scenario/study_id/method_id/family/role/source_type/source_file/source_label/source_line/source_page/source_sha256` 和各指标 `mean/sd/unit/direction/status/missing_reason/source_precision/source_raw`，长表导出将共同来源字段展开到每个指标。额外保留 `drr_star/raw_image_replay/data_access`。真实零、0±0、缺失和结构上不适用分开；原文 `--` 未细分时使用 `not_applicable_or_not_reported`，不擅自判定。主表重复次数和 SD 变异来源未知；MPE/DRR 的 SD 为 null。

- A-Dice 是最终一行的任务平均，不是矩阵整体均值。BWTR 不是普通 BWT，零均值不排除个别旧任务遗忘。
- RMA 分母为独立训练参照，不能用 Non-CL 顺序训练结果代替。RMA/MPE 可大于 1；负 BWTR 与零资源值保持可见。
- E-FWT 仅 Domain，WCD 仅 Class。MiB 的 A-Dice 更高、PLOP 的 WCD 更高；两指标汇总对象不同，差值不是遗忘量。JointTrain 未报告 WCD。
- GPM 保留原论文回放分组，但访问类型为历史子空间，不标成原始图像回放。JointTrain 访问全部任务数据，不参加 CL 冠军或资源前沿；DRR=0 不代表没有历史数据访问。
- SD 只取论文报告值，不生成 CI、显著性检验、箱线图、病例分布或综合总分。
- SAM 行为完成训练阶段，列为测试任务；严格上三角仅为未来域直接测试。缺少随机初始化/独立训练参照，不计算 SAM E-FWT/RMA。阶段曲线只连接实际六点，不补点或平滑；所有派生值注明三位小数输入精度。

## 导出与依赖

新增固定 `altair==6.2.2` 和 `vl-convert-python==1.9.0.post1`（本地 Vega-Lite 6.4 渲染器，[官方项目](https://github.com/vega/vl-convert)）。不通过 CDN 获取渲染器、数据或字体。PNG 使用本机 CJK 字体并检查基本中文字形；找不到字体时明确报错，不输出缺字图。本轮验证字体为系统 Arial Unicode，不随仓库分发。SVG 保留矢量文字，接收机需要可用 CJK 字体；离线 HTML 内嵌 PNG，因此图中文字不依赖接收机安装同一字体。

每次“生成当前视图导出”提供 CSV、JSON、带当前图表的 HTML，以及每张图的 SVG/PNG。JSON 禁止 NaN；CSV 独立 mean/sd，缺失为空且带状态/原因。CSV 对文本公式前缀转义，真实负数保持数值。离线 HTML 内嵌真实图表及来源，无外部资源或脚本。

## 验收与已知边界

完整执行 `python3 -m unittest discover -s tests -v`：**68 项，31.899 秒，OK**。包括全部原测试及新增 9 项测试。见[原始输出](evidence/ch03/unittest.txt)、[浏览器截图与导出](evidence/ch03/README.md)、[导入核对报告](CH03_IMPORT_VERIFICATION.md)。本轮没有修改 TypeScript 或生产前端 bundle，因此没有运行 npm 构建，也不沿用历史 3D 查看器验收结论。

实际启动 Streamlit，在 1440×900、1920×1080、390×844 检查六个视图、场景切换、下载与窄屏回流。验收实例无医学资产且 jobs=0；临时测试 jobs 只存在于自动删除的测试目录。截图来源于实际应用。

原始数据缺口：回放容量缺少可信逐点序列，只展示原图；十种任务顺序没有十次精确观测，只导入明确标注的 SD；主表重复次数及 SD 变异来源未知。没有据此补造数据。

未完成验证：浏览器安全策略阻止打开本地 `file://` 离线报告；没有绕过该限制。HTML 结构、两幅内嵌 PNG 的可读性与非空内容、无外部资源均通过程序校验，但不声称离线 HTML 已在浏览器打开。应用内的 SVG 下载事件已实际确认。控制台无 error；验收期间的 Vega 警告原样留存并在证据索引说明。
