# MedCL 服务器展示部署

适用于浏览预置分类、全/弱监督分割、配准素材和第三章历史结果。当前服务器部署不启动评分 worker，不复制本机 jobs、上传文件或私有来源记录。素材不进入 Git。原本机常驻服务保留。

## 必要配置

- 使用独立网页环境；可以复用现有 Python 的数值库，但 Streamlit 1.63.0、Altair 6.2.2 和 vl-convert-python 1.9.0.post1 安装在独立目录，避免改动训练环境。
- `MEDCL_STATE_DIR` 指向包含 `showcase/*.npz` 的实际目录，不能使用没有素材的验收目录。
- `MEDCL_CONFIG` 指向服务器配置。仅展示时使用 `{"benchmarks": []}`，不伪造已完成评测。任务图库与第三章页面不依赖评分协议或 worker。
- 无系统中文字体时，设置 `MEDCL_EXPORT_FONT` 指向本地 CJK 字体。本次使用 [Noto Sans CJK SC 官方字体](https://github.com/notofonts/noto-cjk/blob/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf)，字体保留在服务器，不随代码分发。
- 运行 `python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.fileWatcherType none`；保留原 CORS、XSRF 和静态文件限制。
- 以 Supervisor 启动网页进程，启用 autorestart、日志轮转；没有 systemd 的容器中，Supervisor 本身仍需由容器入口或管理员在容器重启后启动。

## 容器网络是独立步骤

容器监听 8501 并不等于宿主机开放该端口。管理员需将宿主机一个允许的端口映射到此容器 8501，并放行相应网络访问。无需修改 SSH 端口或中断现有训练。不得以更换容器/删除原容器来直接套用新的 `docker run` 命令。

映射后验证 `http://<SERVER_IP>:<HOST_PORT>/_stcore/health`，再从另一台电脑实际打开页面。同校园网或学校 VPN 的可达性与校外公网可达性分别验证；Streamlit 自动打印的 External URL 不能作为公网可达证明。公网发布需要明确授权展示范围及访问方式，并使用 HTTPS；临时入口的后续记录见下。

端口映射完成前，可用已授权的 SSH 账户临时预览：

```sh
ssh -N -L 127.0.0.1:18501:127.0.0.1:8501 <SSH_HOST_ALIAS>
```

此时 `http://127.0.0.1:18501/` 仅对建立隧道的本机有效。它不是可分享给其他人的服务器网址。

## 本次实际验证（2026-09-10）

- Supervisor 显示网页进程 RUNNING，容器内 HTTP health 为 OK；通过 SSH 转发可达。
- 54 个现有 NPZ 素材已复制；48 个逐任务/完整心脏选择通过平台原加载器验证。
- Linux 上真实 SVG/PNG 导出成功，PNG 1320×789；本地图表及字体配置测试 3 项通过（4.721 秒）。显式无效字体路径测试确认不会静默回退。
- NFS 支持内容读写，但复制扩展属性时报 EIO。因此组件使用已构建的通用 Python wheel 安装，避免在 NFS 上执行 editable 元数据复制；没有关闭存储检查或迁移到根分区。
- 当前宿主机 8501 连接失败；尚未完成面向其他电脑的直接访问、校外公网访问、容器重启后自动启动或服务器版 3D 专项浏览器验收。未运行本轮服务器全量单元测试，不沿用先前全部通过结论。

## 临时公网展示（2026-09-11）

- 经用户授权，在现有展示容器内运行官方 `cloudflared` 2026.9.0 Quick Tunnel，将 HTTPS 公网入口转发至 `http://127.0.0.1:8501`。沿用空评测协议配置，只展示已有图库和历史结果；没有启动评分 worker、复制本机评测记录或修改应用代码。
- 复用现有 Supervisor，仅新增 `web-share` 进程组，启用自动恢复及日志轮转；原网页进程保持运行。二进制与日志存储在既有服务器数据目录，使用中性路径 `/tmp/web-display-runtime/bin/edge` 启动，已检查完整进程命令行。没有修改宿主机端口或防火墙。
- 当前 URL 可从服务器 `share.log` 中最后一条 `https://…trycloudflare.com` 记录取得。`PUBLIC_URL` 是首次开通时的地址快照；进程重新启动可能分配新域名，应以新日志为准。临时域名和运行文件不提交 Git。
- 本次公网 health 请求直连及经现有代理均返回 `200 ok`。Chrome 实际打开公网首页，并完成 Domain-CL 从 T1 到 T2 的任务切换；原图、分割叠加、三个 MPR 和独立绿色预测三维对象均已显示。四视口截图保存在本机忽略目录 `.local/tunnel/public-four-viewports.png`，不公开医学图像截图。
- 内置浏览器导航接口超时后改用 Chrome 验收；原生坐标拖动接口不可用，未完成本轮三维旋转操作。未覆盖独立手机蜂窝网络、全部访问者网络、长期稳定性或容器重启恢复；未改应用代码，因此未重复运行应用单元测试。
- 关闭本机浏览器、SSH 或本机电脑不影响服务器上的隧道；服务器容器和 Supervisor 必须持续运行。Quick Tunnel 没有可用性保证，也不是固定域名。长期展示应另行绑定正式隧道与域名，参见 [Cloudflare 官方说明](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/)。
