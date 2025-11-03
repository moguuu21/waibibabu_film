# PyCinemetrics WebUI（本地版）

运行一个本地 Web UI，通过 HTTP API 在 localhost 上调用现有算法。

## 快速开始（Windows）

- 安装服务器依赖：
  - `pip install -r webui/requirements.txt`
- 启动服务器：
  - 双击 `run_webui.bat` 或在终端中运行 `.\run_webui.bat`
- 在浏览器中打开：`http://127.0.0.1:8000`

## 使用方法

- 方式一：输入完整的本地视频路径，例如 `C:\\videos\\movie.mp4`
- 方式二：拖拽上传或点击选择视频文件。上传完成后会自动填充路径。
- 在 UI 中点击以下任一操作：
  - Shotcut（镜头切换）
  - Colors（色彩）
  - Objects（物体）
  - Subtitles（字幕）
  - ShotScale（镜头景别）
  - Face（人脸相关功能）
- 结果保存到 `img/<video_basename>/` 目录，并在 UI 中预览。

## 目录结构（核心）

- `webui/`
  - `index.html`：主页面（单页应用入口）
  - `styles.css`：样式文件
  - `script.js`：功能脚本（与后端 API 交互）
  - `spa.js`：简易路由/视图切换
  - `video-face-recognition.html`：人脸识别/比对页面
  - `FACE_UI.md`：人脸 UI 使用说明
  - `requirements.txt`：Web 端所需 Python 依赖（Flask 等）
- `src/webserver.py`：Flask 后端服务与 API 实现
- `img/`：分析结果输出目录（按视频基名分子目录）
- `uploads/`：上传的临时文件目录
- `face_database/`：已登记的人脸样本库
- `run_webui.bat`：Windows 启动脚本

## 静态资源与路由

- `GET /` -> `webui/index.html`
- `GET /webui/<file>` -> WebUI 静态资源（HTML/CSS/JS）
- `GET /media/<base>/<file>` -> 结果资源（如导出的图片、SRT 等）
- `GET /face_db/<file>` -> 人脸库图片
- `GET /media-temp/<file>` -> 临时生成的图片

## API 端点

- `GET  /api/results?video_path=...`：列出该视频可用的结果与媒体 URL
- `POST /api/upload`（multipart form-data）
  - 字段：`file`（视频文件）
- `POST /api/shotcut` JSON `{ video_path, th? }`
- `POST /api/colors` JSON `{ video_path, colors_count? }`
- `POST /api/objects` JSON `{ video_path }`
- `POST /api/subtitles` JSON `{ video_path, subtitle_value? }`
- `POST /api/shotscale` JSON `{ video_path }`

人脸相关 API：

- `GET    /api/faces`：列出人脸库
- `POST   /api/faces/add`（multipart form-data）：添加人脸样本
  - 字段：`name`、`file`
- `POST   /api/faces/add_by_path` JSON `{ image_path, name }`：从本地路径添加样本
- `DELETE /api/faces/<filename>`：删除样本
- `POST   /api/face/extract_frames` JSON `{ video_path }`：从关键帧抽取/识别人脸
- `POST   /api/face/compare`（multipart form-data）：人脸比对
  - 字段：`file1`、`file2`

## 输出文件说明（位于 `img/<video_basename>/`）

- `color.png`、`color_palette.png`：主色与调色板
- `objects.png`：物体检测可视化
- `shotscale.png`、`shotscale_timeline.png`：景别统计与时间线
- `subtitles_timeline.png`、`subtitle.srt`：字幕时间线与导出的 SRT
- `predictions_visualization.png`：镜头切分可视化
- `scenes.txt`：场景切分信息
- `frame/`：关键帧目录（按需自动生成）
- `faces/`：抽取的人脸图片与标注

## 配置

- 主机/端口：环境变量 `WEB_HOST`、`WEB_PORT`（默认 `127.0.0.1:8000`）

## 注意事项

- 需要帧的任务若缺少场景关键帧，将通过 TransNetV2 自动提取。
- 物体检测需要 `models/` 目录中的 YOLO 模型文件（本仓库已包含）。
- 所有处理均在本地执行；默认仅监听本机回环地址（不对外开放）。

