# 分割预测三维显示与本轮验收

日期：2026-09-07。仓库 `DLwbm123/medcl`，分支 `codex/medcl-evaluation-platform`。
开始时工作区 HEAD 与远端分支均为 `d7d69ce2894d708f72bf8ef9bb1afeb4665200a5`；未回退基线。
本记录只描述本轮检查，不改写此前验收记录。

## 实现

第四视口采用 **Surface**。三个 MPR 保留 scalar image + Labelmap，分割 3D 从未调用 `setVolumes(image)`，也不使用 MRI preset。配准的 scalar volume / MR-Default 路径保留。

`predictionSurface.ts` 只接收 prediction、标签、预览 spacing 与 origin，没有 image 参数。按 `prediction == label` 分别生成二值网格，使用 vtk.js ImageMarchingCubes 的 0.5 等值面。前景包围盒外增加一个零体素层，使边界对象、单层薄体和全前景体得到闭合表面。未做平滑、连通域删除、补全、重编号或原始数组写入。

每个真实非零标签分别注册 geometry、Surface representation 与 vtkActor。0 不生成表面。MPR 与 Surface 使用相同稳定 LUT；支持非连续 uint16 标签。转换使用预览实际网格：ZYX 数据按 X 最快索引，spacing 转成 XYZ，加入 origin；现有协议仍只接受 identity direction。索引空间提示保留。

Cornerstone core/tools 保持 **5.8.2**。已核对 lockfile、安装的类型/实现，以及 [v5.8.2 Surface 官方示例](https://github.com/cornerstonejs/cornerstone3D/blob/v5.8.2/packages/tools/examples/surfaceRendering/index.ts)。官方示例的原图加载/preset 没有移植到分割 3D。显式声明已经在锁文件中的 `@kitware/vtk.js@36.4.1`，没有版本升级。MarchingCubes 随本地 index.js 打包，不需要新增 worker、WASM 或 CDN；原有 computeWorker bundle 保留。

处理了 5.8.2 的三个实际运行问题：

- uint16 GPU categorical overlay 使用精确保留全部整数值的 Float32 显示副本；prediction 本身不变。验证了 2 / 513。
- `createLocalVolume` 给各切片相同位置：仅在分割挂载期间注册可清理的元数据 provider，提供实际逐层物理位置与 frame。MPR 使用原有 volume clipping 渲染路径，窗口 resize 后叠加仍存在。
- 默认 Surface 更新回调依赖未启用的 PolySeg：对本组件的不可编辑预览 Surface 不安装自动转换更新函数；其他视口仍走原函数。数据改变时重新转换、挂载，未通过伪造转换接口隐瞒错误。

## 控件、空状态和清理

“切面原图”只影响 MPR；“切面叠加”和切面透明度只影响 Labelmap。MPR 使用填充叠加，避免该版本隐藏标签仍残留轮廓的问题。标签选择默认“全部标签”，筛选同时作用于 MPR 和 Surface；切面叠加关闭/重开时保留筛选。

“3D 不透明度”直接控制 vtkActor 材质，不使用 fillAlpha。调整材质不改变相机。Reset 分别恢复 MPR 窗宽窗位/相机与 Surface 材质/相机；3D 相机根据可见预测前景范围定位，并为旋转留出空间。resize 与重新可见不重新加载分割原图。

空状态仅说明“当前预览没有可显示的预测前景”，三个 MPR 仍可用，不推断全分辨率预测为空。Surface 转换/注册/显示异常报告 `PREDICTION_SURFACE_FAILED`、`viewer_ready=false`，清除部分 Surface 并保留 MPR。无 WebGL 等全局失败仍走已有二维静态降级。

关闭/换病例时清理 representations、actors、mapper/polydata、geometry cache、LUT、元数据 provider、工具组、同步器与 DOM observers/listeners。每标签转换前让出事件循环并检查取消状态，显示 await 后再次检查；旧实例使用独立 id，不向新病例挂载。

## 本轮验证

- `npm ci` 成功；`npm run typecheck`、`npm test`（5 文件，24 项）、`npm run build` 成功。
- 仓库根目录 `/opt/miniconda3/bin/python3 -m unittest discover -s tests -v`：56 项通过。系统 `/opt/homebrew/bin/python3` 缺 numpy/h5py，未把其失败冒充通过；使用实际 Streamlit 所在的现有 Conda 环境完成全套测试。
- 几何测试使用真实 MarchingCubes：非对称分离对象、非连续 2/513、世界坐标、闭合边、单层/单体素、全前景、空预测、数组不变与 image-independent 网格/颜色。
- 挂载/控制测试运行 mountViewer 真实分支，使用 GPU/DOM 替身断言 actor 路由、独立透明度、筛选与叠加组合、Reset、取消/重复挂载清理、Surface 失败保留 MPR，以及配准 scalar/preset 路径。
- 实际启动 `streamlit run tests/viewer_acceptance_app.py --server.port=8502`，加载生产 bundle。合成形状为 32×40×48，spacing ZYX = 2.5/1.25/0.75，origin XYZ = 11/-7/3；薄体为 1×40×48。浏览器实际旋转了两个分离对象、薄体、全前景体和单标签对象。
- 实际操作原图/叠加开关、2/7 与 2/513 筛选、0.5 三维不透明度、Reset、病例切换、两次重复挂载、滚动离开/返回，以及 1280×720 → 1100×800 → 恢复尺寸。配准逐项查看 Fixed / Moving / Registered / 两种融合，并操作融合滑杆两端。

生产 index.js 为 5,054,963 字节；8501 平台与 8502 验收页实际服务的文件大小一致，并包含新 Surface 错误码。浏览器在最后构建后重新加载并完成验收，未沿用旧包。

### 原图独立性

固定 prediction、预览几何、全部标签、Reset 后的相机和默认材质，分别用梯度原图、全零原图、棋盘高对比原图运行真实组件。常规应用截图接口输出有损 JPEG，不能把整页 JPEG 压缩误差当作 GPU 差异。合成验收页另有“导出验收画布 PNG”按钮，读取实际 Cornerstone 画布的无损 PNG；该按钮不在产品页面内。

两种强度替换下，3D 的 591,744 个 RGBA 像素均零差异（最大通道差 0）；MPR 分别有 219,696 / 219,317 个像素变化。无损结果见 [image-independence.json](segmentation-3d-evidence/image-independence.json)。复核命令：

```bash
/opt/miniconda3/bin/python3 tests/compare_viewer_screenshots.py
```

### 实际应用截图

所有截图只含合成数据；没有真实患者数据或设计图。

- [四视口](segmentation-3d-evidence/01-four-viewports.jpg)
- [实际旋转后的第二角度](segmentation-3d-evidence/02-rotated.jpg)
- [单标签筛选](segmentation-3d-evidence/03-single-label.jpg)
- [空预测](segmentation-3d-evidence/04-empty-prediction.jpg)
- [uint16 标签 513](segmentation-3d-evidence/05-uint16-label-513.jpg)
- [单层薄体](segmentation-3d-evidence/06-thin-volume.jpg)
- [全前景体](segmentation-3d-evidence/07-full-foreground.jpg)
- [三维不透明度 0.5](segmentation-3d-evidence/08-opacity.jpg)

`registration-*.jpg` 保留配准各图层和融合两端证据。

## 验证边界

浏览器 GPU 验收覆盖本机 Codex 内置浏览器；未做多浏览器/多 GPU 压力测试。故障降级使用自动化注入转换错误验证，未注入真实 GPU context loss。取消与资源清理由控制流测试及实际重复挂载检查，未做 GPU 堆长期泄漏分析。转换在主线程逐标签执行，标签极多时仍可能较慢；没有为了性能删除任何标签。npm ci 的现有依赖审计报告 10 项漏洞，本轮按要求未自动升级或 audit-fix。
