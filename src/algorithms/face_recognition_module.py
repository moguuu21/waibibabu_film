import os
import cv2
import numpy as np
import face_recognition
from PIL import Image, ImageDraw, ImageFont
import shutil
from datetime import datetime
import time
import torch
from pathlib import Path
import sys
# 导入MTCNN
from mtcnn import MTCNN

# 添加项目根目录到系统路径
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入模型
from algorithms.model2 import initialize_face_attribute_model


class FaceRecognition:
    def __init__(self):
        # 初始化MTCNN人脸检测器
        self.detector = MTCNN()

        # 初始化人脸识别模块
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_db_dir = "face_database"

        # 初始化人脸属性识别模型
        self.attr_model = None
        self.attr_names = []
        self.transform = None
        self.device = None
        self._initialize_attribute_model()

        # 确保人脸数据库目录存在
        if not os.path.exists(self.face_db_dir):
            os.makedirs(self.face_db_dir)

        # 加载已知人脸数据库
        self._load_known_faces()

    def _initialize_attribute_model(self):
        """初始化人脸属性识别模型"""
        try:
            project_root = Path(__file__).parent.parent
            model_path = project_root / "algorithms" / "best_attribute_model.pth"

            print(f"尝试从 {model_path} 加载模型...")
            if not model_path.exists():
                print(f"错误: 模型文件 {model_path} 不存在")
                return

            self.attr_model, self.attr_names, self.transform, self.device = \
                initialize_face_attribute_model(project_root, model_path)

            print("模型加载成功")
            print(f"可用属性: {self.attr_names}")
        except Exception as e:
            print(f"模型初始化失败: {str(e)}")

    def _load_known_faces(self):
        """加载已知人脸编码和姓名"""
        # 清空现有数据
        self.known_face_encodings = []
        self.known_face_names = []

        # 检查数据库目录是否存在，不存在则创建
        if not os.path.exists(self.face_db_dir):
            os.makedirs(self.face_db_dir)
            return

        # 读取所有人脸数据
        for filename in os.listdir(self.face_db_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # 提取人名（不含扩展名）
                name = os.path.splitext(filename)[0]

                # 加载图片并编码
                face_image = face_recognition.load_image_file(os.path.join(self.face_db_dir, filename))
                # 使用MTCNN检测人脸
                results = self.detector.detect_faces(face_image)

                # 如果找到人脸，添加到数据库
                if len(results) > 0:
                    # 提取第一个人脸的编码
                    x1, y1, width, height = results[0]['box']
                    x2, y2 = x1 + width, y1 + height
                    face_encodings = face_recognition.face_encodings(face_image, [(y1, x2, y2, x1)])

                    if len(face_encodings) > 0:
                        self.known_face_encodings.append(face_encodings[0])
                        self.known_face_names.append(name)
                        print(f"已加载人脸: {name}")

    def detect_faces(self, image_path):
        """使用MTCNN检测图片中的所有人脸并返回位置"""
        # 加载图片
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 使用MTCNN检测人脸
        results = self.detector.detect_faces(image_rgb)

        # 转换为(顶部, 右侧, 底部, 左侧)格式
        face_locations = []
        for result in results:
            x1, y1, width, height = result['box']
            x2, y2 = x1 + width, y1 + height
            face_locations.append((y1, x2, y2, x1))

        return face_locations, image_rgb

    def recognize_faces(self, image_path):
        """识别图片中的所有人脸"""
        # 加载图片
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 使用MTCNN查找所有人脸位置
        results = self.detector.detect_faces(image_rgb)
        face_locations = []

        for result in results:
            x1, y1, width, height = result['box']
            x2, y2 = x1 + width, y1 + height
            face_locations.append((y1, x2, y2, x1))

        # 提取人脸编码
        face_encodings = face_recognition.face_encodings(image_rgb, face_locations)

        # 转换为OpenCV格式以便绘制
        cv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # 创建结果列表
        face_results = []

        # 循环检查每个面部
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # 与已知人脸比较
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
            name = "未知人物"
            confidence = "N/A"
            attributes = []

            # 使用已知人脸中的欧氏距离最小者
            if len(self.known_face_encodings) > 0:
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                # 计算置信度（距离越小，置信度越高）
                distance = face_distances[best_match_index]
                confidence = round((1 - distance) * 100, 2)

                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            # 如果启用了属性识别，获取人脸属性
            if self.attr_model is not None:
                try:
                    # 裁剪人脸区域
                    face_image = image_rgb[top:bottom, left:right]
                    face_pil = Image.fromarray(face_image)

                    # 预测属性
                    attributes = self._predict_attributes(face_pil)
                except Exception as e:
                    print(f"预测人脸属性时出错: {str(e)}")

            # 存储结果
            face_results.append({
                "location": (top, right, bottom, left),
                "name": name,
                "confidence": confidence,
                "attributes": attributes
            })

            # 在图像上绘制方框
            cv2.rectangle(cv_image, (left, top), (right, bottom), (0, 255, 0), 2)

            # 绘制标签背景
            cv2.rectangle(cv_image, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)

            # 添加名称标签
            cv2.putText(cv_image, f"{name} ({confidence}%)" if confidence != "N/A" else name,
                        (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return cv_image, face_results

    def _predict_attributes(self, face_image):
        """预测人脸属性"""
        if self.attr_model is None:
            print("警告: 属性识别模型未加载")
            return []

        try:
            # 预处理图像
            print(f"输入图像大小: {face_image.size}")
            input_tensor = self.transform(face_image).unsqueeze(0).to(self.device)

            # 预测
            with torch.no_grad():
                output = self.attr_model(input_tensor)
                probs = torch.sigmoid(output).squeeze()
                print(f"原始预测概率: {probs}")

            # 获取预测结果
            predicted_attrs = []
            for i, prob in enumerate(probs):
                if prob > 0.5:  # 降低阈值以显示更多属性
                    print(f"检测到属性: {self.attr_names[i]}, 概率: {prob.item()}")
                    predicted_attrs.append((self.attr_names[i], round(prob.item(), 2)))

            return predicted_attrs
        except Exception as e:
            print(f"属性预测出错: {str(e)}")
            return []

    def add_face(self, image_path, name):
        """添加新人脸到数据库"""
        # 检查人脸数据库目录是否存在
        if not os.path.exists(self.face_db_dir):
            os.makedirs(self.face_db_dir)

        # 加载图片
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 使用MTCNN检测人脸
        results = self.detector.detect_faces(image_rgb)

        if len(results) == 0:
            return False, "未检测到人脸"

        if len(results) > 1:
            return False, "检测到多个人脸，请提供只包含一个人脸的图片"

        # 提取人脸位置和编码
        x1, y1, width, height = results[0]['box']
        x2, y2 = x1 + width, y1 + height
        face_location = (y1, x2, y2, x1)
        face_encodings = face_recognition.face_encodings(image_rgb, [face_location])

        if len(face_encodings) == 0:
            return False, "无法提取人脸特征"

        # 保存人脸图片到数据库
        # 裁剪人脸区域并保存
        top, right, bottom, left = face_location
        face_image = image_rgb[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)

        # 生成唯一文件名（如果同名）
        base_name = name
        filename = f"{name}.jpg"
        counter = 1

        while os.path.exists(os.path.join(self.face_db_dir, filename)):
            filename = f"{base_name}_{counter}.jpg"
            counter += 1

        # 保存图片
        file_path = os.path.join(self.face_db_dir, filename)
        pil_image.save(file_path)

        # 更新内存中的人脸数据库
        self.known_face_encodings.append(face_encodings[0])
        self.known_face_names.append(name)

        return True, f"成功添加人脸: {name}"

    def extract_faces_from_frames(self, frames_dir, output_dir=None):
        """从关键帧中提取所有人脸"""
        if output_dir is None:
            # 使用时间戳创建输出目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(os.path.dirname(frames_dir), f"faces_{timestamp}")

        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        result_faces = []

        # 处理每个关键帧
        for filename in os.listdir(frames_dir):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                frame_path = os.path.join(frames_dir, filename)

                try:
                    # 加载图片
                    image = cv2.imread(frame_path)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # 使用MTCNN检测人脸
                    results = self.detector.detect_faces(image_rgb)

                    # 如果存在人脸
                    if results:
                        # 为每个人脸创建条目并保存裁剪图像
                        for i, result in enumerate(results):
                            x1, y1, width, height = result['box']
                            x2, y2 = x1 + width, y1 + height
                            face_location = (y1, x2, y2, x1)

                            face_image = image_rgb[y1:y2, x1:x2]
                            pil_image = Image.fromarray(face_image)

                            # 保存裁剪的人脸
                            face_filename = f"{os.path.splitext(filename)[0]}_face{i + 1}.jpg"
                            face_path = os.path.join(output_dir, face_filename)
                            pil_image.save(face_path)

                            # 获取人脸编码进行识别
                            face_encoding = face_recognition.face_encodings(image_rgb, [face_location])[0]

                            # 与现有人脸比较
                            name = "未知人物"
                            confidence = "N/A"
                            attributes = []

                            if len(self.known_face_encodings) > 0:
                                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                                face_distances = face_recognition.face_distance(self.known_face_encodings,
                                                                                face_encoding)
                                best_match_index = np.argmin(face_distances)

                                # 计算置信度
                                distance = face_distances[best_match_index]
                                confidence = round((1 - distance) * 100, 2)

                                if matches[best_match_index]:
                                    name = self.known_face_names[best_match_index]

                            # 预测人脸属性
                            if self.attr_model is not None:
                                try:
                                    attributes = self._predict_attributes(pil_image)
                                except Exception as e:
                                    print(f"预测人脸属性时出错: {str(e)}")

                            # 添加到结果
                            result_faces.append({
                                "frame": filename,
                                "face_file": face_filename,
                                "face_path": face_path,
                                "name": name,
                                "confidence": confidence,
                                "location": face_location,
                                "attributes": attributes
                            })

                except Exception as e:
                    print(f"处理帧 {filename} 时出错: {str(e)}")

        return output_dir, result_faces

    def compare_face(self, face_image_path, reference_image_path):
        """比较两张人脸图片是否为同一人"""
        # 加载参考图片
        ref_image = cv2.imread(reference_image_path)
        ref_image_rgb = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        ref_results = self.detector.detect_faces(ref_image_rgb)

        if len(ref_results) == 0:
            return False, "参考图片中未检测到人脸", 0

        # 提取参考人脸编码
        x1, y1, width, height = ref_results[0]['box']
        x2, y2 = x1 + width, y1 + height
        ref_face_location = (y1, x2, y2, x1)
        reference_face_encodings = face_recognition.face_encodings(ref_image_rgb, [ref_face_location])

        if len(reference_face_encodings) == 0:
            return False, "无法提取参考人脸特征", 0

        # 加载要比较的图片
        face_image = cv2.imread(face_image_path)
        face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_results = self.detector.detect_faces(face_image_rgb)

        if len(face_results) == 0:
            return False, "比较图片中未检测到人脸", 0

        # 提取比较人脸编码
        x1, y1, width, height = face_results[0]['box']
        x2, y2 = x1 + width, y1 + height
        face_location = (y1, x2, y2, x1)
        face_encodings = face_recognition.face_encodings(face_image_rgb, [face_location])

        if len(face_encodings) == 0:
            return False, "无法提取比较人脸特征", 0

        # 比较人脸
        matches = face_recognition.compare_faces([reference_face_encodings[0]], face_encodings[0])
        distance = face_recognition.face_distance([reference_face_encodings[0]], face_encodings[0])[0]

        # 计算相似度百分比（距离越小，相似度越高）
        similarity = round((1 - distance) * 100, 2)

        if matches[0]:
            return True, "人脸匹配", similarity
        else:
            return False, "人脸不匹配", similarity

    def create_face_comparison_image(self, image1_path, image2_path, output_path=None):
        """创建两张人脸对比的可视化图片"""
        try:
            # 加载两张图片
            image1 = Image.open(image1_path)
            image2 = Image.open(image2_path)

            # 调整大小，使两张图片高度相同
            height = 300
            width1 = int(image1.width * height / image1.height)
            width2 = int(image2.width * height / image2.height)

            image1 = image1.resize((width1, height))
            image2 = image2.resize((width2, height))

            # 创建新的拼接图像
            total_width = width1 + width2 + 100  # 额外的100像素用于中间的比较结果
            comparison_image = Image.new('RGB', (total_width, height + 50), color=(255, 255, 255))

            # 粘贴两张图片
            comparison_image.paste(image1, (0, 0))
            comparison_image.paste(image2, (width1 + 100, 0))

            # 绘制文本和比较线
            draw = ImageDraw.Draw(comparison_image)

            # 比较人脸
            is_match, message, similarity = self.compare_face(image1_path, image2_path)

            # 确定颜色（匹配为绿色，不匹配为红色）
            color = (0, 180, 0) if is_match else (180, 0, 0)

            # 添加中文支持
            try:
                # 尝试使用系统字体
                font_path = None
                if os.name == 'nt':  # Windows
                    font_path = "C:/Windows/Fonts/simhei.ttf"  # 黑体
                elif os.name == 'posix':  # Linux/Mac
                    # 尝试几个常见的字体路径
                    possible_fonts = [
                        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
                        "/System/Library/Fonts/PingFang.ttc",
                        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
                    ]
                    for p in possible_fonts:
                        if os.path.exists(p):
                            font_path = p
                            break

                # 如果找到了字体，使用它
                if font_path and os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, 16)
                else:
                    # 否则使用默认字体
                    font = ImageFont.load_default()
                    print("警告：未找到中文字体，使用默认字体")
            except Exception as e:
                print(f"加载字体时出错: {str(e)}")
                font = ImageFont.load_default()

            # 添加比较结果
            text = f"相似度: {similarity}%\n{message}"
            draw.text((width1 + 20, height // 2), text, fill=color, font=font)

            # 绘制连接线
            draw.line([(width1, height // 2), (width1 + 100, height // 2)], fill=color, width=2)

            # 保存图像
            temp_dir = os.path.join("img", "temp")
            os.makedirs(temp_dir, exist_ok=True)

            if output_path is None:
                output_path = os.path.join(temp_dir, f"face_comparison_{int(time.time())}.jpg")

            comparison_image.save(output_path)
            return output_path
        except Exception as e:
            print(f"创建人脸比较图像时出错: {str(e)}")
            # 返回一个错误图像
            error_image = Image.new('RGB', (600, 300), color=(255, 255, 255))
            draw = ImageDraw.Draw(error_image)
            draw.text((10, 10), f"错误: {str(e)}", fill=(255, 0, 0))

            temp_dir = os.path.join("img", "temp")
            os.makedirs(temp_dir, exist_ok=True)
            error_path = os.path.join(temp_dir, f"comparison_error_{int(time.time())}.jpg")
            error_image.save(error_path)
            return error_path

    def shotcut_detection(self, v_path=None, image_save=None, frame_save=None, th=0.5):
        """执行镜头切分检测

        Args:
            v_path: 视频文件路径
            image_save: 图像保存目录
            frame_save: 帧保存目录（关键帧将保存到这里）
            th: 检测阈值

        Returns:
            list: 检测到的镜头列表
        """
        if not v_path:
            print("[TransNetV2] 错误: 未提供视频路径")
            return []

        # 确保frame_save目录存在
        if frame_save is None:
            frame_save = os.path.join(image_save, "frame") if image_save else "frame_output"

        os.makedirs(frame_save, exist_ok=True)

        print(f"[TransNetV2] 关键帧将保存到: {frame_save}")

        # 打开视频文件
        cap = cv2.VideoCapture(v_path)
        if not cap.isOpened():
            print(f"[TransNetV2] 错误: 无法打开视频 {v_path}")
            return []

        # 获取视频信息
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[TransNetV2] 视频总帧数: {frame_count}, FPS: {fps}")

        # 预测视频帧
        try:
            video_frames, single_frame_predictions, all_frame_predictions = self.predict_video(v_path)
        except Exception as e:
            print(f"[TransNetV2] 预测视频时出错: {e}")
            cap.release()
            return []

        # 检测场景
        scenes = self.predictions_to_scenes(single_frame_predictions, threshold=th)

        if len(scenes) == 0:
            print("[TransNetV2] 未检测到任何镜头切分")
            cap.release()
            return []

        # 保存关键帧
        saved_frames = []
        frame_len = len(str(frame_count))  # 确定帧号位数

        for i, scene in enumerate(scenes):
            frame_num = scene[0]  # 获取场景开始帧

            # 定位到该帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()

            if ret:
                # 生成文件名
                filename = f"scene_{i + 1}_frame_{frame_num:0{frame_len}d}.jpg"
                save_path = os.path.join(frame_save, filename)

                # 保存帧
                cv2.imwrite(save_path, frame)
                saved_frames.append(save_path)
                print(f"[TransNetV2] 已保存关键帧: {filename}")
            else:
                print(f"[TransNetV2] 警告: 无法读取帧 {frame_num}")

        cap.release()

        # 生成镜头信息列表
        shot_list = []
        for i, scene in enumerate(scenes):
            start_frame = scene[0]
            end_frame = scene[1]
            duration = (end_frame - start_frame) / fps  # 转换为秒

            shot_list.append({
                "shot_id": i + 1,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "duration": duration,
                "keyframe": saved_frames[i] if i < len(saved_frames) else None
            })

        print(f"[TransNetV2] 共保存了 {len(saved_frames)} 个关键帧到 {frame_save}")
        return shot_list
