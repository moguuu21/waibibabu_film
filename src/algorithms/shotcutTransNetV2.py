import os
import numpy as np
import tensorflow as tf
import cv2
import subprocess
import sys
import platform
import sys
import argparse

# 查找系统中的ffmpeg路径
def find_ffmpeg():
    try:
        # 检查本地目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "../.."))
        local_ffmpeg = os.path.join(project_root, "temp", "ffmpeg", "ffmpeg.exe")

        if os.path.exists(local_ffmpeg):
            return local_ffmpeg

        # 检查指定路径
        if platform.system() == "Windows":
            paths = [
                "D:\\FFmeeg\\bin\\ffmpeg.exe",
                "C:\\FFmpeg\\bin\\ffmpeg.exe",
                os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "FFmpeg", "bin", "ffmpeg.exe"),
                os.path.join(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"), "FFmpeg", "bin",
                             "ffmpeg.exe")
            ]
            for path in paths:
                if os.path.exists(path):
                    return path

        # 检查环境变量中的ffmpeg
        try:
            if platform.system() == "Windows":
                result = subprocess.run(["where", "ffmpeg"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            else:
                result = subprocess.run(["which", "ffmpeg"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split("\n")[0]
        except Exception:
            pass

        return "ffmpeg"  # 返回默认命令，依赖PATH环境变量
    except Exception as e:
        print(f"[TransNetV2] Error finding ffmpeg: {e}")
        return "ffmpeg"


# 设置ffmpeg可执行文件的路径
FFMPEG_BIN = find_ffmpeg()
print(f"[TransNetV2] Using FFmpeg: {FFMPEG_BIN}")


class TransNetV2:
    def __init__(self, model_dir=None, use_smoothing=False, window_size=5):
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), "../../models/transnetv2-weights/")
            if not os.path.isdir(model_dir):
                raise FileNotFoundError(f"[TransNetV2] ERROR: {model_dir} is not a directory.")
            else:
                print(f"[TransNetV2] Using weights from {model_dir}.")

        self._input_size = (27, 48, 3)
        self.use_smoothing = use_smoothing
        self.smoothing_window = window_size
        try:
            self._model = tf.saved_model.load(model_dir)
        except OSError as exc:
            raise IOError(f"[TransNetV2] It seems that files in {model_dir} are corrupted or missing. "
                          f"Re-download them manually and retry. For more info, see: "
                          f"https://github.com/soCzech/TransNetV2/issues/1#issuecomment-647357796") from exc

    def predict_raw(self, frames: np.ndarray):
        assert len(frames.shape) == 5 and frames.shape[2:] == self._input_size, \
            "[TransNetV2] Input shape must be [batch, frames, height, width, 3]."
        frames = tf.cast(frames, tf.float32)

        logits, dict_ = self._model(frames)
        single_frame_pred = tf.sigmoid(logits)
        all_frames_pred = tf.sigmoid(dict_["many_hot"])

        return single_frame_pred, all_frames_pred

    def predict_frames(self, frames: np.ndarray):
        assert len(frames.shape) == 4 and frames.shape[1:] == self._input_size, \
            "[TransNetV2] Input shape must be [frames, height, width, 3]."

        def input_iterator():
            # return windows of size 100 where the first/last 25 frames are from the previous/next batch
            no_padded_frames_start = 25
            no_padded_frames_end = 25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)  # 25 - 74

            start_frame = np.expand_dims(frames[0], 0)
            end_frame = np.expand_dims(frames[-1], 0)
            padded_inputs = np.concatenate(
                [start_frame] * no_padded_frames_start + [frames] + [end_frame] * no_padded_frames_end, 0
            )

            ptr = 0
            while ptr + 100 <= len(padded_inputs):
                out = padded_inputs[ptr:ptr + 100]
                ptr += 50
                yield out[np.newaxis]

        predictions = []

        for inp in input_iterator():
            single_frame_pred, all_frames_pred = self.predict_raw(inp)
            predictions.append((single_frame_pred.numpy()[0, 25:75, 0],
                                all_frames_pred.numpy()[0, 25:75, 0]))

            print("\r[TransNetV2] Processing video frames {}/{}".format(
                min(len(predictions) * 50, len(frames)), len(frames)
            ), end="")
        print("")

        single_frame_pred = np.concatenate([single_ for single_, all_ in predictions])
        all_frames_pred = np.concatenate([all_ for single_, all_ in predictions])

        # 应用平滑处理
        if self.use_smoothing:
            single_frame_pred = self.apply_smoothing(single_frame_pred, self.smoothing_window)

        return single_frame_pred[:len(frames)], all_frames_pred[:len(frames)]  # remove extra padded frames

    def apply_smoothing(self, predictions, window_size=5):
        """应用滑动平均滤波平滑预测结果"""
        smoothed = np.zeros_like(predictions)
        half_window = window_size // 2
        for i in range(len(predictions)):
            start = max(0, i - half_window)
            end = min(len(predictions), i + half_window + 1)
            smoothed[i] = np.mean(predictions[start:end])
        return smoothed

    def predict_video(self, video_fn: str):
        # 首先尝试使用ffmpeg
        if os.path.exists(video_fn):
            try:
                import ffmpeg
                print(f"[TransNetV2] 处理视频: {video_fn}")
                print(f"[TransNetV2] 使用FFmpeg路径: {FFMPEG_BIN}")

                # 使用指定的ffmpeg路径
                video_stream, err = (
                    ffmpeg
                    .input(video_fn)
                    .output("pipe:", format="rawvideo", pix_fmt="rgb24", s="48x27")
                    .global_args("-y")  # 允许覆盖输出文件
                    .run(cmd=FFMPEG_BIN, capture_stdout=True, capture_stderr=True)
                )

                video = np.frombuffer(video_stream, np.uint8).reshape([-1, 27, 48, 3])
                return (video, *self.predict_frames(video))

            except Exception as e:
                err = None  # 初始化err变量避免未定义错误
                if hasattr(e, 'stderr') and e.stderr:
                    err = e.stderr
                    err_str = err.decode('utf-8', errors='replace') if isinstance(err, bytes) else str(err)
                    print(f"[TransNetV2] FFmpeg错误输出: {err_str}")

                print(f"[TransNetV2] FFmpeg处理失败: {str(e)}")
                print(f"[TransNetV2] 尝试使用OpenCV替代方案...")
        else:
            print(f"[TransNetV2] 错误: 视频文件不存在 - {video_fn}")

        # 使用OpenCV作为备选方案
        try:
            cap = cv2.VideoCapture(video_fn)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频文件: {video_fn}")

            frames = []
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"[TransNetV2] 使用OpenCV读取视频 - 总帧数: {frame_count}")

            # 进度计数
            processed_frames = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 调整大小为TransNetV2所需的尺寸
                resized_frame = cv2.resize(frame, (48, 27))
                # 转为RGB (OpenCV默认是BGR)
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)

                processed_frames += 1
                if processed_frames % 100 == 0:
                    print(f"\r[TransNetV2] 已处理 {processed_frames}/{frame_count} 帧", end="")

            cap.release()
            print("\n[TransNetV2] 视频读取完成，开始处理帧...")

            if not frames:
                raise ValueError("无法从视频中提取帧")

            video = np.array(frames)
            return (video, *self.predict_frames(video))

        except Exception as e2:
            print(f"[TransNetV2] OpenCV处理失败: {e2}")
            raise ValueError(f"无法处理视频文件: {video_fn}") from e2

    @staticmethod
    def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5):
        predictions = (predictions > threshold).astype(np.uint8)

        scenes = []
        t, t_prev, start = -1, 0, 0
        for i, t in enumerate(predictions):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t
        if t == 0:
            scenes.append([start, i])

        # just fix if all predictions are 1
        if len(scenes) == 0:
            return np.array([[0, len(predictions) - 1]], dtype=np.int32)

        return np.array(scenes, dtype=np.int32)

    @staticmethod
    def visualize_predictions(frames: np.ndarray, predictions):
        from PIL import Image, ImageDraw

        if isinstance(predictions, np.ndarray):
            predictions = [predictions]

        ih, iw, ic = frames.shape[1:]
        width = 25

        # pad frames so that length of the video is divisible by width
        # pad frames also by len(predictions) pixels in width in order to show predictions
        pad_with = width - len(frames) % width if len(frames) % width != 0 else 0
        frames = np.pad(frames, [(0, pad_with), (0, 1), (0, len(predictions)), (0, 0)])

        predictions = [np.pad(x, (0, pad_with)) for x in predictions]
        height = len(frames) // width

        img = frames.reshape([height, width, ih + 1, iw + len(predictions), ic])
        img = np.concatenate(np.split(
            np.concatenate(np.split(img, height), axis=2)[0], width
        ), axis=2)[0, :-1]

        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)

        # iterate over all frames
        for i, pred in enumerate(zip(*predictions)):
            x, y = i % width, i // width
            x, y = x * (iw + len(predictions)) + iw, y * (ih + 1) + ih - 1

            # we can visualize multiple predictions per single frame
            for j, p in enumerate(pred):
                color = [0, 0, 0]
                color[(j + 1) % 3] = 255

                draw.line([x + j, y + 1, x + j, y - int(p * ih * 0.5)], fill=tuple(color), width=1)

        return img

    def shotcut_detection(self, v_path=None, image_save=None, frame_save=None, th=0.5,
                          use_smoothing=False, window_size=5, min_scene_frames=18,
                          solid_variance=0.3, solid_entropy=1.8, brightness_threshold=20):
        """执行镜头切分检测（包含纯色场景过滤）

        Args:
            v_path: 视频文件路径
            image_save: 图像保存目录
            frame_save: 帧保存目录
            th: 检测阈值
            use_smoothing: 是否启用平滑处理
            window_size: 平滑窗口大小
            min_scene_frames: 最小场景帧数，小于此值的场景将被过滤
            solid_variance: 纯色检测的颜色方差阈值
            solid_entropy: 纯色检测的熵阈值
            brightness_threshold: 黑屏检测的亮度阈值（0-255）

        Returns:
            list: 检测到的有效镜头列表
        """

        if not v_path:
            print("[TransNetV2] 错误: 未提供视频路径")
            return []

        # 确保保存目录存在
        if frame_save and not os.path.exists(frame_save):
            os.makedirs(frame_save, exist_ok=True)

        if image_save and not os.path.exists(image_save):
            os.makedirs(image_save, exist_ok=True)

        print(f"[TransNetV2] 开始处理视频 {v_path}")

        # 预测视频帧
        try:
            video_frames, single_frame_predictions, all_frame_predictions = self.predict_video(v_path)
        except Exception as e:
            print(f"[TransNetV2] 预测视频时出错: {e}")
            return []

        # 应用平滑处理（通过参数控制）
        if use_smoothing:
            single_frame_predictions = self.apply_smoothing(single_frame_predictions, window_size)

        # 检测场景
        scenes = self.predictions_to_scenes(single_frame_predictions, threshold=th)

        if len(scenes) == 0:
            print("[TransNetV2] 未检测到任何镜头切分")
            return []

        # 过滤短场景
        if min_scene_frames > 0:
            scene_durations = scenes[:, 1] - scenes[:, 0]
            valid_indices = np.where(scene_durations >= min_scene_frames)[0]
            scenes = scenes[valid_indices]

            if len(scenes) == 0:
                print(f"[TransNetV2] 过滤后未检测到有效场景（所有场景帧数 < {min_scene_frames}）")
                return []
            else:
                print(f"[TransNetV2] 过滤掉 {len(scene_durations) - len(scenes)} 个短场景")

        # 提取关键帧并保存（增加纯色检测）
        if frame_save:
            # 打开视频
            cap = cv2.VideoCapture(v_path)
            if not cap.isOpened():
                print(f"[TransNetV2] 错误: 无法打开视频 {v_path}")
                return []

            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            print(f"[TransNetV2] 视频总帧数: {frame_count}")
            frame_len = len(str((int)(frame_count)))
            valid_shots = []
            start = 0

            # 保存每个场景的关键帧
            for scene in scenes:
                scene_start = scene[0]
                scene_end = scene[1]

                # 设置视频位置到场景开始帧
                cap.set(cv2.CAP_PROP_POS_FRAMES, scene_start)
                ret, frame = cap.read()

                if ret:
                    # 检测是否为纯色场景
                    is_solid = self.is_solid_color_frame(frame, solid_variance, solid_entropy, brightness_threshold)

                    if not is_solid:
                        # 生成文件名并保存帧
                        filename = os.path.join(frame_save, f"frame{scene_start:0{frame_len}d}.png")
                        cv2.imwrite(filename, frame)
                        valid_shots.append([start, scene_start, scene_start - start])
                        start = scene_start
                        print(f"[TransNetV2] 保存关键帧: {filename}")
                    else:
                        print(f"[TransNetV2] 过滤纯色关键帧: 帧 {scene_start}")
                else:
                    print(f"[TransNetV2] 警告: 无法读取帧 {scene_start}")

            # 保存最后一个场景的结束帧（如果需要）
            if scenes[-1][1] < frame_count:
                cap.set(cv2.CAP_PROP_POS_FRAMES, scenes[-1][1])
                ret, frame = cap.read()
                if ret:
                    # 检测结束帧是否为纯色
                    is_solid = self.is_solid_color_frame(frame, solid_variance, solid_entropy, brightness_threshold)
                    if not is_solid:
                        filename = os.path.join(frame_save, f"frame{scenes[-1][1]:0{frame_len}d}.png")
                        cv2.imwrite(filename, frame)

            cap.release()

            print(f"[TransNetV2] 保存了 {len(valid_shots)} 个有效关键帧到 {frame_save}")
            return valid_shots
        else:
            # 如果没有提供frame_save，只返回场景信息（增加纯色检测逻辑）
            valid_scenes = []
            for i in range(len(scenes)):
                start = scenes[i][0]
                end = scenes[i][1]

                # 快速检测场景起始帧是否为纯色（简化逻辑）
                cap = cv2.VideoCapture(v_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                ret, frame = cap.read()
                cap.release()

                if ret and not self.is_solid_color_frame(frame, solid_variance, solid_entropy, brightness_threshold):
                    valid_scenes.append([start, end, end - start])

            print(f"[TransNetV2] 检测到 {len(valid_scenes)} 个有效镜头切分")
            return valid_scenes

    def is_solid_color_frame(self, frame, variance_threshold=0.3, entropy_threshold=1.8, brightness_threshold=20):
        """检测帧是否为纯色场景"""
        if frame is None:
            return True

        # 计算颜色方差
        variances = [np.var(frame[:, :, i]) for i in range(3)]
        avg_variance = np.mean(variances)

        # 计算颜色熵
        entropies = []
        for i in range(3):
            hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
            valid = hist[hist > 0]
            if len(valid) == 0:
                entropies.append(0)
                continue
            p = valid / np.sum(valid)
            entropies.append(-np.sum(p * np.log2(p)))
        avg_entropy = np.mean(entropies)

        # 计算平均亮度
        brightness = np.mean(frame)

        # 判断是否为纯色：低方差+低熵，或极亮/极暗
        is_solid = (avg_variance < variance_threshold and avg_entropy < entropy_threshold) or \
                   (brightness < brightness_threshold) or \
                   (brightness > 220)

        return is_solid


def getFrame_number(f_path):
    f = open(f_path, 'r')
    Frame_number = []

    i = 0
    for line in f:
        NumList = [int(n) for n in line.split()]
        Frame_number.append(NumList[1])

    print(Frame_number)
    return Frame_number


def transNetV2_run(v_path, image_save, th, use_smoothing=False, window_size=5):
    """
    运行TransNetV2分镜头检测并保存结果

    Args:
        v_path: 视频文件路径
        image_save: 结果保存目录
        th: 检测阈值
        use_smoothing: 是否启用平滑处理
        window_size: 平滑窗口大小
    """
    # 确保输出目录存在
    if not os.path.exists(image_save):
        os.makedirs(image_save)

    frame_save = os.path.join(image_save, "frame")
    if not os.path.exists(frame_save):
        os.makedirs(frame_save)
    else:
        # 清理旧文件
        try:
            for f in os.listdir(frame_save):
                file_path = os.path.join(frame_save, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        except Exception as e:
            print(f"[TransNetV2] 无法清理目录: {e}")

    # 初始化模型（启用平滑处理参数）
    model = TransNetV2(use_smoothing=use_smoothing, window_size=window_size)

    file = v_path
    if os.path.exists(file + ".predictions.txt") or os.path.exists(file + ".scenes.txt"):
        print(f"[TransNetV2] {file}.predictions.txt or {file}.scenes.txt already exists. "
              f"Skipping video {file}.", file=sys.stderr)
        return []

    # 预测视频帧
    try:
        video_frames, single_frame_predictions, all_frame_predictions = model.predict_video(file)
    except Exception as e:
        print(f"[TransNetV2] 预测视频时出错: {e}")
        return []

    # 应用平滑处理（通过参数控制）
    if use_smoothing:
        single_frame_predictions = model.apply_smoothing(single_frame_predictions, window_size)

    # 检测场景
    scenes = model.predictions_to_scenes(single_frame_predictions, threshold=th)
    np.savetxt(os.path.join(image_save, "scenes.txt"), scenes, fmt="%d")

    # 可视化预测结果
    try:
        pil_image = model.visualize_predictions(
            video_frames, predictions=(single_frame_predictions, all_frame_predictions)
        )
        pil_image.save(os.path.join(image_save, "predictions_visualization.png"))
    except Exception as e:
        print(f"[TransNetV2] 可视化预测结果时出错: {e}")

    # 提取关键帧编号
    try:
        frame_numbers = getFrame_number(os.path.join(image_save, "scenes.txt"))
        if frame_numbers:
            frame_numbers.pop()  # 移除最后一个多余的帧号（根据原逻辑）
    except Exception as e:
        print(f"[TransNetV2] 提取帧号时出错: {e}")
        frame_numbers = []

    # 打开视频并保存关键帧
    cap = cv2.VideoCapture(v_path)
    if not cap.isOpened():
        print(f"[TransNetV2] 错误: 无法打开视频 {v_path}")
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[TransNetV2] 视频总帧数: {frame_count}")
    frame_len = len(str(frame_count))
    shot_len = []
    start_frame = 0

    # 保存第一帧
    ret, frame = cap.read()
    if not ret:
        print("[TransNetV2] 错误: 无法读取第一帧")
        cap.release()
        return []

    first_frame_path = os.path.join(frame_save, f"frame{'0' * (frame_len - len(str(0)))}{0}.png")
    cv2.imwrite(first_frame_path, frame)
    shot_len.append([start_frame, 0, 0 - start_frame])
    start_frame = 0

    # 保存分镜关键帧
    for frame_num in frame_numbers:
        frame_num = frame_num + 1  # 调整帧号（根据原逻辑）
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            print(f"[TransNetV2] 警告: 无法读取帧 {frame_num}")
            continue

        frame_id = f"frame{'0' * (frame_len - len(str(frame_num)))}{frame_num}.png"
        output_path = os.path.join(frame_save, frame_id)
        cv2.imwrite(output_path, frame)
        shot_len.append([start_frame, frame_num, frame_num - start_frame])
        start_frame = frame_num

    cap.release()
    print(f"[TransNetV2] 处理完成，已保存 {len(shot_len)} 个关键帧到 {frame_save}")
    return shot_len