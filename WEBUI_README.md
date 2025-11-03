# PyCinemetrics WebUI (本地版)

运行一个本地 Web UI，通过 HTTP API 在 localhost 上调用现有算法。

## 快速开始 (Windows)

- 安装服务器依赖：
  - `pip install -r webui/requirements.txt`
- 启动服务器：
  - 双击 `run_webui.bat` 或在终端中运行 `.\run_webui.bat`
- 在浏览器中打开：`http://127.0.0.1:8000`

## 使用方法

- 方式一：输入完整的本地视频路径，例如 `C:\\videos\\movie.mp4`
- 方式二：直接将视频文件拖拽到输入框下方的“拖拽上传”区域，或点击该区域选择文件。上传完成后会自动填充路径。
- 点击其中一个操作：Shotcut（镜头切换）、Colors（色彩）、Objects（物体）、Subtitles（字幕）、ShotScale（镜头景别）
- 结果保存在 `img/<video_basename>/` 目录下，并在 UI 中预览

## API 端点

- `POST /api/shotcut` JSON `{ video_path, th? }`
- `POST /api/colors` JSON `{ video_path, colors_count? }`
- `POST /api/objects` JSON `{ video_path }`
- `POST /api/subtitles` JSON `{ video_path, subtitle_value? }`
- `POST /api/shotscale` JSON `{ video_path }`
- `GET  /api/results?video_path=...` -> 列出可用的结果和媒体 URL

## 配置

- 主机/端口：设置环境变量 `WEB_HOST`、`WEB_PORT`（默认为 `127.0.0.1:8000`）

## 注意事项

- 服务器重用 `src/algorithms` 中的模块。对于需要帧的任务，如果缺少场景关键帧，将通过 TransNetV2 提取。
- 物体检测需要 `models/` 目录下的 YOLO 文件（本仓库中已包含）。
- 所有处理都在本地进行；服务器不提供远程访问。
