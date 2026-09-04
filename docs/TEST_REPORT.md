# MedCL 三维评测平台工程验收 · 2026-09-04

结论：外部审阅中与毕业论文平台范围一致的建议已落地。页面保持“分割 / 分类 / 配准 → 增量场景 → 监督方式 → 具体协议”的纵向结构；分割和体数据配准接入统一、只读的 Cornerstone3D 单病例查看器。数值评分仍由独立 Python worker 完成，查看器不训练、不配准、不重采样，也不接触隐藏真值。

基线为 `710fdfab29dc93c0b9597b43f1b2c50a4f4b8388`，开发分支为 `codex/medcl-evaluation-platform`。提交前工作区已有与本项目无关的 `.hatch-pet/`、`.pet-runs/` 未跟踪目录；本轮未修改、未纳入提交。

## 运行环境

- Python 3.12.9；Streamlit 1.63.0、NumPy 2.2.3、h5py 3.16.0、safetensors 0.5.3。
- Node.js 25.3.0、npm 11.7.0。
- Cornerstone core/tools 5.8.2、Streamlit component-v2-lib 0.2.0、TypeScript 5.9.2、Vite 7.1.7、Vitest 3.2.4；`events` 3.3.0 与 `url` 0.11.4 是浏览器兼容依赖。
- 生产环境读取仓库内的预构建静态包，不需要 Node 服务或 CDN。修改 TypeScript/CSS 后必须重新 build。

## 自动化测试

实际执行：

```text
PATH=/opt/miniconda3/bin:$PATH python3 -m unittest discover -s tests -v
Ran 35 tests in 14.587s
OK
```

结果为 **35 passed，0 failed，0 errors，0 skipped**。覆盖既有指标、队列、报告、沙箱和页面提交，以及新增的标签空间/`label_shift`、输出头 allow-list、体数据配准资产、可选 registered/warped 数组、几何默认值、三维预览边界、旧二维降级和 envelope 恶意输入。

前端从 lockfile 干净安装后实际执行：

```text
npm ci --no-audit --prefer-offline
npm run typecheck
npm test -- --run
Test Files 2 passed (2)
Tests 8 passed (8)
npm run build
2000 modules transformed
built in 5.48s
```

最终 typecheck、Vitest 和 Vite build 均为退出码 0。bundle 含 `index.js` 与本地 compute worker；生产页面没有远程脚本依赖。

## 浏览器 smoke test

在干净的本地浏览器会话中实际打开运行中的 Streamlit 页面，并检查：

1. 一级入口是分割、分类、配准；结果页仍显示矩阵、客户端聚合、病例与报告页签。
2. 分割合成非对称体显示 Z/Y/X 三个 MPR 和一个 3D viewport，最终构建的 4 个 Cornerstone canvas 均实际调整为 `982×526` backing pixels；原图/预测分别开关、prediction labelmap 叠加、类别选择、透明度、病例 Dice、滚轮切片和病例切换可用。
3. 配准合成非对称体显示 fixed、moving、registered、fixed+moving、fixed+registered、融合滑杆、可选 warped prediction 和 TRE；最终构建的 4 个 canvas 均为 `982×494` backing pixels。
4. W/L、Pan、Zoom、Crosshair、Reset、叠加开关和透明度控件均可操作。切换病例会清理并重建 rendering engine/volume/segmentation，不复用旧体数据。
5. 缺 registered 的配准记录明确显示 Fixed + Moving，并提示 TRE 仍来自 predicted landmarks；空分割预测保留原图 MPR，明确说明没有前景 overlay/3D labelmap。
6. 干净会话无控制台 error。Cornerstone 在仅 2 个 Z 切片的合成分割体上记录过一条非致命的最近切片匹配 warning；四视口、切片和覆盖层仍正常渲染。

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
