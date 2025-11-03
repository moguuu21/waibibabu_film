import os
import threading
from pathlib import Path
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify, send_from_directory, Response
from werkzeug.utils import secure_filename

# Algorithms
from algorithms.shotcutTransNetV2 import TransNetV2
from algorithms.colorAnalyzer import ColorAnalyzer
from algorithms.objectDetection import ObjectDetection
from algorithms.subtitleEasyOcr import SubtitleProcessor
from algorithms.shotscale import ShotScale
from algorithms.face_recognition_module import FaceRecognition


BASE_DIR = Path(__file__).resolve().parent.parent
IMG_DIR = BASE_DIR / "img"
WEBUI_DIR = BASE_DIR / "webui"
UPLOAD_DIR = BASE_DIR / "uploads"
FACE_DB_DIR = BASE_DIR / "face_database"


def ensure_dirs(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def get_video_base(video_path: str) -> str:
    return Path(video_path).name.rsplit('.', 1)[0]


def ensure_keyframes(video_path: str, target_dir: Path) -> None:
    """Ensure target_dir/frame has keyframes extracted. Uses TransNetV2 for scene-based frames."""
    frame_dir = target_dir / "frame"
    frame_dir.mkdir(parents=True, exist_ok=True)

    # If frames already exist, skip
    has_any = any(p.suffix.lower() in {".jpg", ".jpeg", ".png"} for p in frame_dir.glob("*"))
    if has_any:
        return

    # Use TransNetV2 to extract frames (also saves artifacts into target_dir)
    model = TransNetV2()
    model.shotcut_detection(
        v_path=video_path,
        image_save=str(target_dir),
        frame_save=str(frame_dir),
        th=0.5,
    )


def list_results(video_path: str) -> Dict[str, Any]:
    base = get_video_base(video_path)
    out_dir = IMG_DIR / base
    results: Dict[str, Any] = {
        "base": base,
        "dir": str(out_dir),
        "exists": out_dir.exists(),
        "files": {},
    }
    if not out_dir.exists():
        return results

    files = {
        "color": "color.png",
        "color_palette": "color_palette.png",
        "objects": "objects.png",
        "shotscale": "shotscale.png",
        "shotscale_timeline": "shotscale_timeline.png",
        "subtitles_timeline": "subtitles_timeline.png",
        "subtitle_srt": "subtitle.srt",
        "predictions_visualization": "predictions_visualization.png",
        "scenes": "scenes.txt",
    }
    for key, rel in files.items():
        p = out_dir / rel
        if p.exists():
            results["files"][key] = f"/media/{base}/{rel}"
    return results


app = Flask(__name__)
ensure_dirs(IMG_DIR)
ensure_dirs(UPLOAD_DIR)
ensure_dirs(FACE_DB_DIR)

# Lazy-init face module (heavy load). Reuse singleton across requests.
_FACE: Optional[FaceRecognition] = None


def get_face() -> FaceRecognition:
    global _FACE
    if _FACE is None:
        _FACE = FaceRecognition()
    return _FACE

ALLOWED_VIDEO_EXTS = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.m4v', '.webm'}


@app.get("/")
def index() -> Response:
    index_path = WEBUI_DIR / "index.html"
    if not index_path.exists():
        return Response("WebUI not found. Please create webui/index.html.", status=404)
    return send_from_directory(str(WEBUI_DIR), "index.html")


@app.get("/webui/<path:filename>")
def webui_static(filename: str):
    return send_from_directory(str(WEBUI_DIR), filename)


@app.get("/media/<base>/<path:filename>")
def media_file(base: str, filename: str):
    directory = IMG_DIR / base
    return send_from_directory(str(directory), filename)


@app.get("/face_db/<path:filename>")
def face_db_file(filename: str):
    return send_from_directory(str(FACE_DB_DIR), filename)


@app.get("/media-temp/<path:filename>")
def media_temp(filename: str):
    temp_dir = IMG_DIR / "temp"
    ensure_dirs(temp_dir)
    return send_from_directory(str(temp_dir), filename)


@app.get("/api/results")
def api_results():
    video_path = request.args.get("video_path", "")
    if not video_path:
        return jsonify({"ok": False, "error": "missing video_path"}), 400
    return jsonify({"ok": True, "data": list_results(video_path)})


@app.post("/api/upload")
def api_upload():
    file = request.files.get('file')
    if file is None or file.filename == '':
        return jsonify({"ok": False, "error": "no file uploaded"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTS:
        return jsonify({"ok": False, "error": f"unsupported file type: {ext}"}), 400

    filename = secure_filename(file.filename)
    save_path = UPLOAD_DIR / filename
    if save_path.exists():
        stem = save_path.stem
        i = 1
        while True:
            candidate = UPLOAD_DIR / f"{stem}-{i}{ext}"
            if not candidate.exists():
                save_path = candidate
                break
            i += 1

    try:
        file.save(str(save_path))
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    return jsonify({
        "ok": True,
        "message": "upload success",
        "data": {
            "filename": save_path.name,
            "saved_path": str(save_path),
        }
    })


@app.post("/api/shotcut")
def api_shotcut():
    data = request.get_json(force=True, silent=True) or {}
    video_path = data.get("video_path")
    th = float(data.get("th", 0.5))
    if not video_path or not Path(video_path).exists():
        return jsonify({"ok": False, "error": "Invalid video_path"}), 400

    base = get_video_base(video_path)
    out_dir = IMG_DIR / base
    ensure_dirs(out_dir)
    frame_dir = out_dir / "frame"
    ensure_dirs(frame_dir)

    try:
        model = TransNetV2()
        scenes = model.shotcut_detection(
            v_path=video_path,
            image_save=str(out_dir),
            frame_save=str(frame_dir),
            th=th,
        )
        return jsonify({
            "ok": True,
            "message": f"Shotcut done. Scenes: {len(scenes) if scenes else 0}",
            "results": list_results(video_path),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/api/colors")
def api_colors():
    data = request.get_json(force=True, silent=True) or {}
    video_path = data.get("video_path")
    colors_count = int(data.get("colors_count", 5))
    if not video_path or not Path(video_path).exists():
        return jsonify({"ok": False, "error": "Invalid video_path"}), 400

    base = get_video_base(video_path)
    out_dir = IMG_DIR / base
    ensure_dirs(out_dir)

    try:
        ensure_keyframes(video_path, out_dir)
        analyzer = ColorAnalyzer(str(out_dir))
        analyzer.analyze_colors(colors_count)
        return jsonify({
            "ok": True,
            "message": "Color analysis done",
            "results": list_results(video_path),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/api/objects")
def api_objects():
    data = request.get_json(force=True, silent=True) or {}
    video_path = data.get("video_path")
    if not video_path or not Path(video_path).exists():
        return jsonify({"ok": False, "error": "Invalid video_path"}), 400

    base = get_video_base(video_path)
    out_dir = IMG_DIR / base
    ensure_dirs(out_dir)

    try:
        ensure_keyframes(video_path, out_dir)
        detector = ObjectDetection(str(out_dir))
        detector.object_detection()
        return jsonify({
            "ok": True,
            "message": "Object detection done",
            "results": list_results(video_path),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/api/subtitles")
def api_subtitles():
    data = request.get_json(force=True, silent=True) or {}
    video_path = data.get("video_path")
    subtitle_value = int(data.get("subtitle_value", 48))
    if not video_path or not Path(video_path).exists():
        return jsonify({"ok": False, "error": "Invalid video_path"}), 400

    base = get_video_base(video_path)
    out_dir = IMG_DIR / base
    ensure_dirs(out_dir)

    try:
        processor = SubtitleProcessor()
        subtitle_str, subtitle_list = processor.getsubtitleEasyOcr(
            video_path, str(out_dir), subtitle_value
        )
        processor.subtitle2Srt(subtitle_list, str(out_dir))
        return jsonify({
            "ok": True,
            "message": "Subtitle OCR done",
            "results": list_results(video_path),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/api/shotscale")
def api_shotscale():
    data = request.get_json(force=True, silent=True) or {}
    video_path = data.get("video_path")
    if not video_path or not Path(video_path).exists():
        return jsonify({"ok": False, "error": "Invalid video_path"}), 400

    base = get_video_base(video_path)
    out_dir = IMG_DIR / base
    ensure_dirs(out_dir)

    try:
        ensure_keyframes(video_path, out_dir)
        ss = ShotScale(str(out_dir))
        ss.shotscale_recognize()
        return jsonify({
            "ok": True,
            "message": "Shot scale analysis done",
            "results": list_results(video_path),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# -------------------- Face APIs --------------------
@app.get("/api/faces")
def api_faces_list():
    items = []
    for p in sorted(FACE_DB_DIR.glob("*")):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        items.append({
            "filename": p.name,
            "name": p.stem,
            "url": f"/face_db/{p.name}",
        })
    return jsonify({"ok": True, "faces": items})


@app.post("/api/faces/add")
def api_faces_add():
    try:
        name = request.form.get('name', '').strip()
        file = request.files.get('file')
        if not name:
            return jsonify({"ok": False, "error": "缺少姓名 name"}), 400
        if file is None or file.filename == '':
            return jsonify({"ok": False, "error": "未上传图片文件"}), 400

        # Save temp upload, then add via recognizer (which crops and stores)
        filename = secure_filename(file.filename)
        tmp_path = UPLOAD_DIR / f"upload_face_{filename}"
        file.save(str(tmp_path))
        try:
            face = get_face()
            ok, msg = face.add_face(str(tmp_path), name)
            if not ok:
                return jsonify({"ok": False, "error": msg}), 400
            # Reload in-memory DB to ensure new face available
            face._load_known_faces()
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

        return jsonify({"ok": True, "message": f"已添加人脸样本：{name}"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/api/faces/add_by_path")
def api_faces_add_by_path():
    data = request.get_json(force=True, silent=True) or {}
    image_path = str(data.get("image_path", "")).strip()
    name = str(data.get("name", "")).strip()
    if not image_path or not name:
        return jsonify({"ok": False, "error": "缺少 image_path 或 name"}), 400
    p = Path(image_path)
    if not p.exists() or not p.is_file():
        return jsonify({"ok": False, "error": "image_path 无效"}), 400
    try:
        face = get_face()
        ok, msg = face.add_face(str(p), name)
        if not ok:
            return jsonify({"ok": False, "error": msg}), 400
        face._load_known_faces()
        return jsonify({"ok": True, "message": f"已添加：{name}"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.delete("/api/faces/<path:filename>")
def api_faces_delete(filename: str):
    p = FACE_DB_DIR / filename
    if not p.exists() or not p.is_file():
        return jsonify({"ok": False, "error": "文件不存在"}), 404
    try:
        p.unlink()
        # refresh in-memory DB
        try:
            get_face()._load_known_faces()
        except Exception:
            pass
        return jsonify({"ok": True, "message": "已删除"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/api/face/extract_frames")
def api_face_extract_frames():
    data = request.get_json(force=True, silent=True) or {}
    video_path = data.get("video_path")
    if not video_path or not Path(video_path).exists():
        return jsonify({"ok": False, "error": "Invalid video_path"}), 400

    base = get_video_base(video_path)
    out_dir = IMG_DIR / base
    ensure_dirs(out_dir)
    frames_dir = out_dir / "frame"
    ensure_dirs(frames_dir)
    try:
        # ensure keyframes exist
        ensure_keyframes(video_path, out_dir)
        # output directory for faces (stable path)
        faces_dir = out_dir / "faces"
        ensure_dirs(faces_dir)
        # clean old faces
        for fp in faces_dir.glob("*.jpg"):
            try:
                fp.unlink()
            except Exception:
                pass
        fr = get_face()
        out_dir_real, faces = fr.extract_faces_from_frames(str(frames_dir), str(faces_dir))
        items = []
        for it in faces:
            face_file = it.get("face_file")
            items.append({
                "frame": it.get("frame"),
                "face_file": face_file,
                "url": f"/media/{base}/faces/{face_file}",
                "name": it.get("name"),
                "confidence": it.get("confidence"),
                "attributes": it.get("attributes", []),
                "path": str((faces_dir / face_file).resolve()),
            })
        return jsonify({
            "ok": True,
            "message": f"已从关键帧提取人脸：{len(items)}",
            "data": {
                "faces": items,
                "dir": str(faces_dir),
            }
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/api/face/compare")
def api_face_compare():
    file1 = request.files.get('file1')
    file2 = request.files.get('file2')
    if not file1 or not file2:
        return jsonify({"ok": False, "error": "请上传两张图片 file1 与 file2"}), 400
    tmp1 = UPLOAD_DIR / f"cmp_{secure_filename(file1.filename)}"
    tmp2 = UPLOAD_DIR / f"cmp_{secure_filename(file2.filename)}"
    file1.save(str(tmp1))
    file2.save(str(tmp2))
    try:
        fr = get_face()
        is_match, message, similarity = fr.compare_face(str(tmp1), str(tmp2))
        out_path = fr.create_face_comparison_image(str(tmp1), str(tmp2))
        out_name = Path(out_path).name
        return jsonify({
            "ok": True,
            "match": bool(is_match),
            "message": message,
            "similarity": similarity,
            "image_url": f"/media-temp/{out_name}",
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    finally:
        for p in (tmp1, tmp2):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass


def main():
    host = os.environ.get("WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("WEB_PORT", "8000"))
    # threaded=True allows concurrent requests while long tasks run
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
