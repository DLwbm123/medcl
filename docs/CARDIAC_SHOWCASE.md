# 心脏七前景示例

当前入口为全监督示例 → Class-CL → T3 →「查看最终七类完整心脏示例」，见 [逐任务示例](CONTINUAL_TASK_SHOWCASE.md)。新的弱监督逐任务素材使用 255 表示未知，兼容下述旧素材的未知值 4；下文保留当时实现与验收记录。

2026-09-07，起始本地/远端 SHA：`f1ca697d761743ca8b8966ee8f66abbdd0590526`。

入口：主页 → 全监督示例 → 示例病例「心脏 · 七前景」→ 三维浏览。前列腺示例继续保留。心脏使用类增量最终阶段所用完整七类 H5 的首个病例原始标签，属于独立参考展示，不产生模型推理、评测作业或评分。

预览保留标签 0–7，不重编号。平台按原始编号展示，不根据任务文件名推断每个整数的器官名称。原始 256×256×150（切片轴在末尾）转成 ZYX 后，按 1/2/2 步长采样为 150×128×128；全部七个前景仍存在。H5 没有物理 spacing/origin/direction，保留 index-space 提示，不声称真实毫米比例或患者方向。

三个 MPR 保留影像与标签叠加；第四视口继续使用 prediction-only、逐标签生成的 Surface。静态切片、MPR 和 Surface 使用一致的七种颜色；第七类新增黄色，避免与第一类重复。弱监督涂鸦的未标注值 4 继续不着色。

## 素材准备

在有 h5py/nibabel 的已有环境运行：

```sh
python scripts/prepare_showcase.py --cardiac /private/path/whole_heart_test.h5 > /private/path/cardiac.tar
```

私有归档只包含 `segmentation-cardiac.npz` 和 `cardiac-provenance.private.json`；解包到管理员的 `$MEDCL_STATE_DIR/showcase/`（默认 `~/.local/state/medcl/showcase/`）。NPZ 和来源记录为 owner-only。素材不随 Git 发布。源 H5 只读，没有修改训练、测试或评分文件，也没有把标签加入评测传输白名单。

## 本轮验证

- `npm ci`、`npm run typecheck`、`npm test`（25 项）、`npm run build` 成功，生产 `build/index.js` 已更新。依赖版本及 lockfile 不变。
- Python 全套测试通过（57 项）；包括七类素材加载、原始标签保持、第一病例边界、缺失类别拒绝、静态七色叠加及示例导航。测试不创建评分作业。
- 真实 Streamlit 8501 加载生产包：心脏七类分别筛选并检查，每类有独立的非空立体对象；已旋转整体，检查第三类内部对象与第七类表面。MPR 筛选颜色对应。
- 合成 Streamlit 8502 新增七个独立前景（1–7），实际旋转并执行 Reset。公开截图见 [七色合成渲染](cardiac-showcase-evidence/synthetic-seven-rotated.jpg)。真实心脏截图仅私有保存，未发布。
- 前端测试验证七类独立网格范围、七种不同颜色，并重新通过 image-independence 几何/材质测试和配准路径回归。没有将上轮逐像素比较或全部浏览器矩阵计为本轮重跑。

限制：七类同时不透明时内部结构会被遮挡，可选择单类或调节「3D 不透明度」。网格阶梯保留；未做平滑、删小对象或器官补全。尚无原始物理几何可用于解剖方向/毫米比例核验。
