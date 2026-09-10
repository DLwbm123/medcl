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

映射后验证 `http://<SERVER_IP>:<HOST_PORT>/_stcore/health`，再从另一台电脑实际打开页面。同校园网或学校 VPN 的可达性与校外公网可达性分别验证；Streamlit 自动打印的 External URL 不能作为公网可达证明。公网发布还需要已授权的入口、访问控制和 HTTPS，当前未开通。

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
