# 第三章仪表板实际验收证据

2026-09-10；实际 Streamlit 应用 `http://127.0.0.1:8501/`，进入“方法比较”。启动实例无医学资产、无 worker，数据库 jobs=0。图片为浏览器真实截图，无设计图、医学病例或静态 HTML 替代。

## 自动化测试

命令：`python3 -m unittest discover -s tests -v`，使用已有 Python 3.12.9 环境。结果：**68 tests / 31.899s / OK**；[原始输出](unittest.txt)。覆盖原平台测试、新数据/图表/页面/导出测试。新增 9 项测试中包含 15 种具体图形的真实 SVG/PNG 渲染与图像非空检查。未运行 npm / TypeScript 构建（本轮无对应源码改动）。

## 截图索引

1440×900、1920×1080、390×844 均做实际页面检查。DOM 测量整页 scrollWidth 等于 viewport 宽度；移动端图表单列回流，完整方法名、中文、负号、误差线、色标可读。表格允许内部横向滚动。六个子视图有桌面及移动端代表截图。

| 截图 | 实际像素 | 内容 |
|---|---|---|
| [capability-1920.jpg](capability-1920.jpg) | 1920×1080 | 能力指标 |
| [capability-mobile.jpg](capability-mobile.jpg) | 390×844 | 能力指标 |
| [capacity-original-1920.jpg](capacity-original-1920.jpg) | 1920×1080 | 回放容量原图 |
| [class-overview-1440.jpg](class-overview-1440.jpg) | 1440×900 | Class 总览 |
| [class-wcd-1440.jpg](class-wcd-1440.jpg) | 1440×900 | Class A-Dice / WCD |
| [cross-scenario-1920.jpg](cross-scenario-1920.jpg) | 1920×1080 | 跨场景缺失值 |
| [cross-scenario-mobile.jpg](cross-scenario-mobile.jpg) | 390×844 | 跨场景缺失值 |
| [data-sources-1920.jpg](data-sources-1920.jpg) | 1920×1080 | 数据与来源 |
| [data-sources-mobile.jpg](data-sources-mobile.jpg) | 390×844 | 数据与来源 |
| [downloads-mobile.jpg](downloads-mobile.jpg) | 390×844 | 实际下载按钮 |
| [organ-overview-1440.jpg](organ-overview-1440.jpg) | 1440×900 | Organ 总览 |
| [overview-1920-charts.jpg](overview-1920-charts.jpg) | 1920×1080 | 总览 |
| [overview-1920.jpg](overview-1920.jpg) | 1920×1080 | 总览 |
| [overview-mobile-chart.jpg](overview-mobile-chart.jpg) | 390×844 | 总览 |
| [overview-mobile-top.jpg](overview-mobile-top.jpg) | 390×844 | 总览 |
| [resources-1920.jpg](resources-1920.jpg) | 1920×1080 | 资源权衡 |
| [robustness-1920.jpg](robustness-1920.jpg) | 1920×1080 | 任务顺序 SD |
| [robustness-mobile.jpg](robustness-mobile.jpg) | 390×844 | 任务顺序 SD |
| [sam-1920.jpg](sam-1920.jpg) | 1920×1080 | SAM 双矩阵与差值 |
| [sam-mobile.jpg](sam-mobile.jpg) | 390×844 | SAM 双矩阵与差值 |
| [sam-stage-T4-1920.jpg](sam-stage-T4-1920.jpg) | 1920×1080 | T4 六阶段观测 |

## 页面与隔离检查

- 浏览器实际切换 Domain/Class/Organ；Class 保留 MiB / PLOP 排序差异，Organ DER++ 未报告；SAM 的 T4 曲线首次学习标记正确。
- AppTest 覆盖场景切换清理选择、方法筛选、参照、重置、空选择及六视图；0/1 jobs 均能使用历史页，2 jobs 时原平台比较与不相容阻止逻辑保留。测试 jobs 仅在临时目录中生成，自动删除。
- 首页/3D Surface/评分/worker/隐私边界代码没有修改，原回归测试通过；本次未重新执行 3D 专项浏览器验收。

## 导出文件

- [CSV](overview-report.csv)：UTF-8 BOM，独立 mean/sd 和完整来源；真实负值保留数值。
- [JSON](overview-report.json)：严格 JSON，null 与零分开，包含当前筛选范围及来源。
- [离线 HTML](overview-report.html)：总览两幅实际 PNG 图及当前数据/来源/指标/协议，无 CDN、脚本或外部资源。
- [A-Dice SVG](Domain-CL-A-Dice.svg)、[A-Dice PNG](Domain-CL-A-Dice.png)、[环图 SVG](方法数量构成.svg)、[环图 PNG](方法数量构成.png)。
- HTMLParser 解析两幅内嵌图，尺寸分别 1320×789 / 1320×517；Pillow 解码并检验图像非空。SVG XML 有效；PNG 中文、完整标签及误差线已视觉查看。实际应用导出按钮生成下载项，浏览器确认 SVG download 事件。

## 控制台与未完成验证

[控制台采集](browser-console.json) 保留验收过程日志，error 列表为空。Vega 在 Streamlit 注入数据前可能记录 `Infinite extent` 的短暂 warning；最终数据、坐标轴及图表显示正常。采集中曾出现的合并图例/联合排序 warning 已通过统一图例与显式排序修复，未删除历史采集。最终顺序图已重新截图。没有将 warning 冒称为零日志。

离线 HTML 的**浏览器打开验收未完成**：工具安全策略阻止 `file://` 地址，没有尝试间接提供地址绕过。结构、内嵌图像与无外部资源的程序校验已通过；这不能替代该浏览器打开验收。没有宣称所有手工验证完成。

原始数据缺口与逐项导入核对见[核对报告](../../CH03_IMPORT_VERIFICATION.md)；实现与启动见[模块文档](../../CH03_VISUALIZATION.md)。
