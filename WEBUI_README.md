# PyCinemetrics WebUI 运行指南

PyCinemetrics WebUI 基于 Flask 提供网页端交互，调用 `src/` 下的分析能力；桌面端（Qt）和 WebUI 各自使用独立虚拟环境，互不干扰。

---

## 目录速览
- `run_webui.bat`：Windows 启动脚本（创建/复用 `.webui-venv`，安装依赖并启动 `src/webserver.py`）
- `.webui-venv/`：WebUI 专用虚拟环境（可随项目一起打包分发）
- `.venv/`：桌面端/全量分析的虚拟环境
- `webui/`：前端静态文件 (`index.html`, `styles.css`, `script.js`, `spa.js`, `video-face-recognition.html` 等)
- `src/webserver.py`：Web API 入口
- `img/`：生成的分析截图/可视化
- `uploads/`：WebUI 上传的视频文件
- `face_database/`：人脸库
- 其他数据/模型：`models/`, `resources/`, `native/` 等

---

## 环境与依赖
- 桌面端/全功能分析：使用 `.venv`，依赖清单 `requirements.txt`（含 PySide6、TensorFlow、检测/识别等全量依赖）。
- WebUI：使用 `.webui-venv`，依赖清单 `webui/requirements.txt`（Flask + 必需依赖，已补充 `numpy` 以避免 “No module named numpy” 报错）。
- 互通性：桌面端环境可运行 WebUI；WebUI 的轻量环境不一定能跑桌面端。

---

## 快速上手

### 1) 准备
确保安装 Python 3.8+ 且 `python`/`pip` 在 PATH。

### 2) 桌面端 / 全量分析
```bat
cd C:\01subject\03design\01do\movie\pyCinemetrics
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python src/main.py
```

### 3) WebUI（自动处理 `.webui-venv`）
```bat
cd C:\01subject\03design\01do\movie\pyCinemetrics
.\run_webui.bat
```
- 首次运行会创建/检查 `.webui-venv` 并安装 `webui/requirements.txt`。
- 指定 Python：
```bat
set WEBUI_PYTHON=C:\Path\To\python.exe
.\run_webui.bat
```
- 跳过依赖检查（已预装时）：
```bat
set WEBUI_SKIP_INSTALL=1
.\run_webui.bat
```
- 自定义 Host/Port：
```bat
set WEB_HOST=0.0.0.0
set WEB_PORT=8000
.\run_webui.bat
```
启动后访问 <http://127.0.0.1:8000>

---

## 常见问题 & 排查
- **No module named numpy**：`webui/requirements.txt` 已包含 `numpy`。若仍报错，执行：
  ```bat
  cd C:\01subject\03design\01do\movie\pyCinemetrics
  .\.webui-venv\Scripts\activate
  pip install numpy==1.24.3
  ```
  或删除 `.webui-venv` 后重跑 `run_webui.bat`。
- **Failed to create .webui-venv**：确认 Python 3.8+ 在 PATH，或设置 `WEBUI_PYTHON` 使用指定解释器。
- **依赖安装慢/失败**：可在 `webui/requirements.txt` 中切换可信镜像源；或预先装好 `.webui-venv` 与项目一并分发。
- **端口未生效**：检查启动日志 `Running on http://...`；需在运行前设置 `WEB_HOST/WEB_PORT`。
- **虚拟环境损坏**：删除对应虚拟环境目录（`.webui-venv` 或 `.venv`），按上方步骤重新创建。

---

## API 简要
| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 返回 webui/index.html |
| GET | `/webui/<file>` | 前端静态文件 |
| GET | `/media/<base>/<file>` | 分析产物（含 CSV/SRT 等） |
| GET | `/face_db/<file>` | 人脸库文件 |
| GET | `/media-temp/<file>` | 临时媒体 |
| GET | `/api/results?video_path=...` | 返回分析结果 |
| POST | `/api/upload` | 上传视频 (`multipart/form-data`, `file`) |
| POST | `/api/shotcut` `/api/colors` `/api/objects` `/api/subtitles` `/api/shotscale` | 分析各模块，JSON 需带 `video_path` |
| GET | `/api/faces` | 列出人脸库 |
| POST | `/api/faces/add` | 上传文件添加人脸 (`name` + `file`) |
| POST | `/api/faces/add_by_path` | 文件路径添加人脸 (`image_path` + `name`) |
| DELETE | `/api/faces/<filename>` | 删除人脸 |
| POST | `/api/face/extract_frames` | 从视频抽帧建库 |
| POST | `/api/face/compare` | 人脸比对 |

