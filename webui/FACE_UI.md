# 人脸识别功能（WebUI）

本地 WebUI 中新增了“人脸识别”板块，包含以下能力：

- 样本库管理：上传人脸样本至 `face_database/`，列出与删除已有样本
- 关键帧提取人脸：基于前序“镜头切分”的关键帧，批量检测/裁剪人脸并展示
- 以图比图：上传两张图片进行人脸对比，返回相似度与匹配结论，并生成可视化对比图

## API 端点

- `GET  /api/faces` → 列出样本库
- `POST /api/faces/add` (multipart: `name`, `file`) → 添加样本
- `POST /api/faces/add_by_path` JSON `{ image_path, name }` → 从服务器路径添加样本（如关键帧提取的人脸裁剪图）
- `DELETE /api/faces/<filename>` → 删除样本
- `POST /api/face/extract_frames` JSON `{ video_path }` → 从关键帧提取人脸，保存到 `img/<base>/faces/`
- `POST /api/face/compare` (multipart: `file1`, `file2`) → 比较两张人脸，返回可视化对比图 `/media-temp/...`

静态资源：
- 样本库图片：`/face_db/<filename>`
- 比较生成图片：`/media-temp/<filename>`

## 依赖

人脸能力需要安装如下包（示例）：

```
pip install mtcnn face_recognition pillow numpy opencv-python
```

注：`face_recognition` 依赖 dlib，请确保你的环境已准备好 dlib（Windows 可使用预编译 wheel）。

