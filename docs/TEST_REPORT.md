# MedCL 首页视觉保真修正验收 · 2026-09-06

结论：已按 `medcl-homepage-fidelity-fix` 增量包原样接入首页模块、医学背景、脑形 Logo、统一图标、CSS 和测试，没有重新设计或把医学背景替换为纯渐变。开始基线为 `f5d2c23d264768f931896c6cb06ef9dcc2f86d5a`，分支为 `codex/medcl-evaluation-platform`。

## 应用与测试

更新器先以 `--check --diff` 验证 `app.py` 和 `tests/test_app.py` 的 Git blob 与审阅基线一致，再以 `--apply` 写入。原文件备份位于 Git 私有目录 `.git/medcl-ui-backups/`，不进入版本控制；交付包本身也没有复制到仓库。

完整平台测试：

```text
/opt/miniconda3/bin/python -m unittest discover -s tests -v
Ran 55 tests in 15.519s
OK
```

其中新增 16 项首页模块测试；包外的 13 项更新器契约测试另行执行，结果为 `Ran 13 tests in 0.063s · OK`。两组合计覆盖包声明的 29 项针对性检查，但不替代上述完整平台回归。

## 实际浏览器验收

使用独立临时 `MEDCL_STATE_DIR` 实际运行 `python3 run.py --no-browser`，没有读取或写入日常数据库。浏览器确认：

1. 医学背景、脑形 Logo、四个导航图标、三类任务图标、四项概览图标和病例示意均正确加载；没有退回纯渐变或椭圆占位图。
2. “开始评测”进入任务中心的三类任务起点；分类任务卡进入类别增量流程；四项导航均为原生可操作控件。
3. 正常模式显示真实空状态和实时零值，不混入合成记录。另在隔离开发模式创建两条明确标记的合成验收记录，首页“查看”准确打开所点击的分割记录。
4. 1600×900、1920×1080、1366×768、390×844 的实际 Streamlit 页面均无横向溢出；桌面保持左文右图和三列任务，窄屏保持背景并按单列自然滚动。
5. 本地 CSS、SVG data URI 和 WebP 资产均来自固定白名单；测试确认没有 HTTP/CDN/外部字体引用。

实际页面截图：

- `docs/screenshots/homepage-fidelity-1600x900.png`
- `docs/screenshots/homepage-fidelity-1920x1080.png`
- `docs/screenshots/homepage-fidelity-1366x768.png`
- `docs/screenshots/homepage-fidelity-390x844.png`

截图为正常模式的实际 Streamlit 页面，只显示真实空状态；不包含真实医学影像、患者信息、私有路径或测试数据库。评分、协议、隐私、worker、沙箱和 Cornerstone TypeScript 均未修改。

---

# MedCL 首页展示重构验收 · 2026-09-06

结论：依据已批准的首页参考图完成了克制的展示层重构。平台现在默认进入首页，使用宽屏浅色外壳、真实导航、左文右图主视觉、三类任务入口、最近评测和实时平台概览；医学影像查看器仍为深色工作区。首页矩阵与病例图只使用本地 CSS/SVG，并明确标为功能示意，不写入评测记录或统计。指标、评分、上传、沙箱、数据协议和 Cornerstone TypeScript 均未修改。

本轮开始 HEAD 为 `c425595dc09bc9e9482d2dd4d7c4a71cb8fb7b31`，分支为 `codex/medcl-evaluation-platform`。工作区已有 `.hatch-pet/`、`.pet-runs/` 未跟踪目录，本轮未修改或纳入提交。系统 Python 为 3.14.6；项目测试使用现有 `/opt/miniconda3` Python 3.12 环境；Node.js 为 25.3.0，npm 为 11.7.0。

## 自动化与浏览器验收

实际执行：

```text
/opt/miniconda3/bin/python -m unittest discover -s tests -v
Ran 39 tests in 15.452s
OK
```

新增回归覆盖首页默认导航、两个主行动按钮、首页最近记录的准确跳转、统一视觉变量，以及只有最终阶段时不渲染空指标卡。既有 39 项测试继续覆盖提交、worker、指标、报告、安全边界和查看器协议。

在本地运行的 Streamlit 页面中完成以下交互验收：

1. “开始评测”进入任务中心并重置到三类任务入口；首页任务卡可直接进入对应任务；“查看评测记录”和最近记录按钮打开正确记录。
2. 连续点击已选中的“域增量”仍保持选中，不再出现页面异常。
3. 1600×900、1920×1080、1366×768 和 390×844 四种视口均无横向溢出；宽屏显示左文右图和三列任务，窄屏隐藏装饰图并将内容顺序回流。
4. 首页首屏使用当前协议和可见记录实时计算概览；无真实协议时显示真实空状态，不用合成数值冒充结果。
5. 浏览器资源检查未发现外部图片、字体或脚本请求。正常模式隐藏合成任务；开发模式仅用于合成工程验收。
6. 合成多阶段分割记录可显示 Final average、BWT、Forgetting、阶段矩阵和病例查看器。查看器返回 Ready，三向 MPR、预测叠加和可拖拽 3D viewport 均可见；二维预览在 3D 就绪时默认收起。

截图保存在 `docs/screenshots/`：

- `homepage-1600x900.png`
- `homepage-1920x1080.png`
- `homepage-1366x768.png`
- `homepage-mobile-390x844.png`
- `task-center-synthetic.png`
- `submission-synthetic.png`
- `result-synthetic.png`
- `viewer-synthetic.png`

后四张只包含程序生成的合成工程验收数据；没有患者信息、真实医学影像、隐藏标签、私有路径或服务器配置。由于本轮未修改 `medcl_cornerstone/frontend/src`，按仓库约定没有重复构建预构建 bundle。小体积的两层合成 volume 在 Cornerstone 切换病例时仍会产生非阻断的切片定位 warning；查看器保持 Ready，真实三维数据协议与既有渲染逻辑未受本轮影响。

---

# MedCL 评测平台 UI/UX 重构验收 · 2026-09-04

结论：平台已形成宽屏、浅色、数据优先的答辩展示界面。首页任务横向排列，提交设置分为双列，结果页以核心指标、阶段矩阵、病例可视化和横向方法比较为主。浅灰文字和大面积彩色提示已收敛；Cornerstone3D 保持深色医学影像工作区并随主容器拉伸。评分、安全和数据协议未修改。

本轮用户给定基线为 `6c5611d7698bb421c55043671212abf91b4939c7`；实际开始 HEAD 为该基线的直接后继 `bef2c3e163e4c3d8529f40080278f913cd6d023b`。开发分支为 `codex/medcl-evaluation-platform`，本轮未回退、rebase 或改写历史。工作区已有与本项目无关的 `.hatch-pet/`、`.pet-runs/` 未跟踪目录；本轮未修改、未纳入提交。
开始前系统 `python3 --version` 为 Python 3.14.6；项目回归继续使用已有的 Python 3.12.9 环境。

## 运行环境

- Python 3.12.9；Streamlit 1.63.0、NumPy 2.2.3、h5py 3.16.0、safetensors 0.5.3。
- Node.js 25.3.0、npm 11.7.0。
- Cornerstone core/tools 5.8.2、Streamlit component-v2-lib 0.2.0、TypeScript 5.9.2、Vite 7.1.7、Vitest 3.2.4；`events` 3.3.0 与 `url` 0.11.4 是浏览器兼容依赖。
- 生产环境读取仓库内的预构建静态包，不需要 Node 服务或 CDN。修改 TypeScript/CSS 后必须重新 build。

## 自动化测试

实际执行：

```text
PATH=/opt/miniconda3/bin:$PATH python3 -m unittest discover -s tests -v
Ran 37 tests in 15.210s
OK
```

结果为 **37 passed，0 failed，0 errors，0 skipped**。新增检查锁定集中式视觉变量、`1760px` 宽屏上限和三列任务入口；正常/开发模式、提交、指标、队列、报告、沙箱和三维查看器回归继续通过。另使用两条相容已完成记录实际渲染方法比较，得到方法为行、T1/T2/T3 为列的宽表，无页面异常。

固定基线的前端验收记录如下（本轮未修改 Cornerstone TypeScript 或预构建 bundle）：

```text
npm ci --no-audit --prefer-offline
npm run typecheck
npm test -- --run
Test Files 3 passed (3)
Tests 15 passed (15)
npm run build
2001 modules transformed
built in 6.13s
```

最终 typecheck、Vitest 和 Vite build 均为退出码 0。bundle 含 `index.js` 与本地 compute worker；生产页面没有远程脚本依赖。

## 浏览器 smoke test

在本地浏览器会话中实际打开运行中的 Streamlit 页面，并检查：

1. 1920×1080 下主容器从约 `x=120` 延伸至 `x=1800`，三类任务在同一行展示；1280 及更窄视口正常回流。无产品层级编号、章节编号、顶栏服务状态或普通界面开发信息。
2. 评测提交页在宽屏上将“评测设置”和“提交内容”并排；训练条件改为窄条浅蓝说明，非关键提示改为简短次要文字。
3. 结果页的五项持续学习指标在 1920宽屏上同行展示；大面积绿/黄背景已改为白色细边框和小量语义色文字。阶段矩阵使用统一蓝色色阶，病例和客户端表头以中文显示。
4. 方法比较使用两条相容记录实际渲染；宽表为“方法 / 来源 / T1 / T2 / T3”，配套横向分组条形图。
5. 使用临时、确定性生成的非医学体数据验收 Cornerstone3D：在 1280 宽浏览器中查看器实测宽 `1189px`，占满主容区；Z/Y/X 三个 MPR、3D volume、工具栏、融合、overlay 和 TRE 均显示正常。组件仍使用深色工作区，本轮未改动 TypeScript 渲染逻辑。
6. `MEDCL_SHOW_DEMOS=1` 仍可完成合成协议提交、评分、报告下载和非法上传错误展示；正常模式仍隐藏合成、待接入和不可用项。
7. 页面重载后监测 8 秒，无浏览器控制台 error 或 warning。宽屏截图、临时体数据、配置和状态目录均未写入仓库。

这些是合成工程验收，不是论文方法性能。真实医学影像截图未保存到仓库。

## 私有 GPU 服务器真实数据标签回放

按用户要求，在用户指定的私有 GPU 服务器上对一个真实前列腺分割测试 HDF5 做只读端到端评分。读取到 72 张 `256×256` 切片和 3 个完整病例；把测试分割标签原样作为“假定模型输出”，经正式提交校验、匿名 ID 对齐、worker 病例聚合、私有预览读取和 viewer envelope 编解码后得到：

| 检查项 | 实际结果 |
|---|---|
| 队列状态 | completed |
| 全局 Foreground Dice | **1.0** |
| Final average | **1.0** |
| 私有三维预览 | 3 个 `segmentation-volume` |
| 首个预览显示网格 | `24×128×128` ZYX |
| envelope volumes | `image`, `prediction` |
| 权限 | 状态/任务目录 0700；结果/预览 0600 |

Dice 1.0 是标签恒等回放的必然结果，只证明真实 HDF5 读取、标签/样本对齐、病例 Dice、三维预览和浏览器数据桥一致；它**不代表模型推理、泛化能力、论文实验或临床性能**。服务器原始数据未修改；测试临时目录及其中的预测、结果和预览已经删除。没有将服务器地址、账号、路径、病例 ID、真实体数据、标签或截图写入仓库。

## 关键修复与边界

- 分割标签在整数截断后应用非背景 `label_shift`，并验证实际值只能属于 `{0} ∪ classes`；域/类/任务场景分别执行对应标签空间约束。
- `allowed_output_heads` 由协议集中校验，页面和后端同时强制；不能通过伪造请求开启未登记输出头。
- 三维预览每阶段、每任务最多 3 例；每轴最多 128、总体最多 1,000,000 voxel，确定性下采样并同步 spacing。无 spacing 使用 `[1,1,1]` 索引空间；无完整方向信息不声称病人方向。
- 体数据配准的管理员资产是同形 fixed/moving `[N,Z,Y,X]` 和 moving/fixed points `[N,K,3]`。registered 与 warped prediction 由提交者在可信环境外部生成并对齐 fixed grid，只用于定性显示；TRE 仍是主指标。
- 浏览器信封白名单限制 schema、volume 名称/顺序、dtype、shape、offset、长度和总大小；拒绝空洞、重叠、截断、尾随数据及非法 labelmap。
- hidden segmentation truth、fixed points 和 fixed truth segmentation 不进入预览或浏览器；报告/aggregate 不嵌入体数据。普通资产冻结仍是 size/mtime 轻量检查，不是密码学内容冻结。

## 已知限制与待接入项

- Cornerstone 查看器是论文范围内的只读单病例质控界面，不是 ITK-SNAP 替代品；没有编辑、测量、DICOM patient orientation 复原、任意 affine、形变场计算或配准重采样。
- 当前分割三维显示使用 Cornerstone `Labelmap` representation，并把同一 representation 挂到 MPR 与 3D viewport；没有预计算或传输独立 surface mesh。这样保留体素标签、支持类别过滤，也避免另一套 mesh 协议。超大体会按上述上限生成有损预览，不能作为原始体归档。
- 配准真实 volume/registered/warped 契约已实现，但仍待接入经确认的真实固定网格、坐标和合规外部预测；真实配准模型效果未测试。
- 分类仍聚焦类别增量与逻辑客户端联邦评分模拟；没有联邦训练、通信或安全聚合。研究模型可提交合规预测，任意上传 Python 模型、UNet/EfficientNet/SAMCL 直接执行仍未开放。
- 合成、未训练基线和本次真值回放都不能补齐论文实验；真实方法结论必须另行提供冻结协议、合法模型预测和可复现实验。
