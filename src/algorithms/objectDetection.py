import cv2
import os
import numpy as np
import csv
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import logging
import requests
from tqdm import tqdm
import urllib.request

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ObjectDetection")


class ObjectDetection:
    def __init__(self, path):
        """初始化物体检测类

        Args:
            path: 视频帧保存的路径
        """
        self.path = path
        self.frame_dir = os.path.join(path, "frame")
        self.result_csv = os.path.join(path, "objects.csv")
        self.result_image = os.path.join(path, "objects.png")

        # YOLO模型文件路径设置为本地目录
        self.models_dir = os.path.abspath(
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models"))
        self.config_path = os.path.join(self.models_dir, "yolov3.cfg")
        self.weights_path = os.path.join(self.models_dir, "yolov3.weights")
        self.coco_path = os.path.join(self.models_dir, "coco.names")

        # 检查和创建frame目录
        if not os.path.exists(self.frame_dir):
            if os.path.exists(path):
                os.makedirs(self.frame_dir, exist_ok=True)
                logger.info(f"创建frame目录: {self.frame_dir}")
            else:
                logger.error(f"路径不存在: {path}")
                raise FileNotFoundError(f"路径不存在: {path}")

        # 检查模型目录是否存在
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir, exist_ok=True)
            logger.info(f"创建模型目录: {self.models_dir}")

    def _check_model_files(self):
        """检查模型文件是否存在

        Returns:
            missing_files: 缺失的模型文件列表
        """
        # 检查配置文件
        missing_files = []
        if not os.path.exists(self.config_path):
            missing_files.append("yolov3.cfg")

        # 检查权重文件
        if not os.path.exists(self.weights_path):
            missing_files.append("yolov3.weights")

        # 检查类别名称文件
        if not os.path.exists(self.coco_path):
            missing_files.append("coco.names")

        return missing_files

    def _load_yolo_model(self):
        """加载YOLO模型

        Returns:
            net: YOLO网络模型
            output_layers: 输出层
            classes: 类别名称列表
        """
        # 检查模型文件是否存在
        missing_files = self._check_model_files()
        if missing_files:
            error_msg = f"缺少YOLO模型文件: {', '.join(missing_files)}，请确保模型文件已下载到 {self.models_dir} 目录"
            logger.error(f"加载YOLO模型失败: {error_msg}")
            raise FileNotFoundError(error_msg)

        try:
            # 读取coco.names中的类别
            with open(self.coco_path, 'r') as f:
                classes = [line.strip() for line in f.readlines()]

            # 加载YOLO网络
            net = cv2.dnn.readNet(self.weights_path, self.config_path)

            # 获取输出层
            layer_names = net.getLayerNames()
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

            return net, output_layers, classes
        except Exception as e:
            logger.error(f"加载YOLO模型时出错: {str(e)}")
            raise

    def _detect_objects(self, image_path):
        """检测图像中的物体

        Args:
            image_path: 图像文件路径

        Returns:
            detections: 检测到的物体列表，每个元素包含类别和置信度
        """
        try:
            # 加载模型
            net, output_layers, classes = self._load_yolo_model()

            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"无法读取图像: {image_path}")
                return []

            height, width, _ = image.shape

            # 预处理图像
            blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)

            # 前向传播
            outs = net.forward(output_layers)

            # 解析结果
            class_ids = []
            confidences = []
            boxes = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    # 仅保留高置信度的检测结果
                    if confidence > 0.5:
                        # 目标边界框坐标
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # 矩形坐标
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # 非极大值抑制
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            detections = []
            for i in indices:
                if isinstance(i, list):  # 兼容不同版本的OpenCV
                    i = i[0]

                box = boxes[i]
                class_id = class_ids[i]
                confidence = confidences[i]

                detections.append({
                    'class': classes[class_id],
                    'confidence': confidence,
                    'box': box
                })

            return detections
        except Exception as e:
            logger.error(f"检测物体时出错 {image_path}: {str(e)}")
            return []

    def object_detection(self):
        """执行物体检测

        Returns:
            results: 包含每个帧检测到的物体信息的列表
        """
        try:
            # 检查模型文件是否存在
            missing_files = self._check_model_files()
            if missing_files:
                logger.error(f"缺少必要的模型文件: {', '.join(missing_files)}，无法进行物体检测")
                logger.info(f"请确保以下文件存在于 {self.models_dir} 目录:")
                logger.info("1. yolov3.cfg - YOLO模型配置文件")
                logger.info("2. yolov3.weights - YOLO模型权重文件")
                logger.info("3. coco.names - COCO数据集类别名称文件")
                return None

            # 检查图像目录是否存在
            if not os.path.exists(self.frame_dir):
                logger.error(f"图像目录不存在: {self.frame_dir}")
                return None

            # 获取所有图像文件
            image_files = [f for f in os.listdir(self.frame_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

            if not image_files:
                logger.warning(f"图像目录中没有图像文件: {self.frame_dir}")
                return None

            # 检测每个图像中的物体
            results = []

            logger.info(f"开始检测物体，共 {len(image_files)} 张图像")

            for i, image_file in enumerate(sorted(image_files)):
                if i % 10 == 0:  # 每10张图像记录一次日志
                    logger.info(f"正在处理第 {i + 1}/{len(image_files)} 张图像")

                # 提取帧ID
                frame_id = os.path.splitext(image_file)[0].replace("frame", "")

                # 完整的图像路径
                image_path = os.path.join(self.frame_dir, image_file)

                # 检测物体
                detections = self._detect_objects(image_path)

                # 收集结果
                for detection in detections:
                    results.append({
                        'frame_id': frame_id,
                        'class': detection['class'],
                        'confidence': detection['confidence']
                    })

            # 保存结果到CSV文件
            self._save_results_to_csv(results)

            # 生成统计图表
            self._generate_statistics(results)

            logger.info(f"物体检测完成，检测到 {len(results)} 个物体")

            return results
        except Exception as e:
            logger.error(f"执行物体检测时出错: {str(e)}")
            return None

    def _save_results_to_csv(self, results):
        """保存结果到CSV文件

        Args:
            results: 包含每个帧检测到的物体信息的列表
        """
        try:
            with open(self.result_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['frame_id', 'class', 'confidence'])

                for result in results:
                    writer.writerow([
                        result['frame_id'],
                        result['class'],
                        result['confidence']
                    ])

            logger.info(f"结果已保存到CSV文件: {self.result_csv}")
        except Exception as e:
            logger.error(f"保存结果到CSV文件时出错: {str(e)}")

    def _generate_statistics(self, results):
        """生成物体检测统计图表

        Args:
            results: 包含每个帧检测到的物体信息的列表
        """
        try:
            if not results:
                logger.warning("没有检测结果，无法生成统计图表")
                return

            # 统计每个类别的数量
            class_counts = {}
            for result in results:
                class_name = result['class']
                if class_name in class_counts:
                    class_counts[class_name] += 1
                else:
                    class_counts[class_name] = 1

            # 按数量排序
            sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

            # 只取前15个最常见的类别，避免图表过于拥挤
            if len(sorted_classes) > 15:
                sorted_classes = sorted_classes[:15]
                logger.info(f"图表仅显示前15个最常见的物体类别，共 {len(class_counts)} 个类别")

            # 准备绘图数据
            class_names = [item[0] for item in sorted_classes]
            counts = [item[1] for item in sorted_classes]

            # 创建图表
            plt.figure(figsize=(12, 8))

            # 创建条形图
            plt.subplot(1, 2, 1)
            bars = plt.barh(class_names, counts, color='#1976D2')

            # 添加数值标签
            for bar, count in zip(bars, counts):
                plt.text(
                    bar.get_width() + 0.3,
                    bar.get_y() + bar.get_height() / 2,
                    str(count),
                    va='center'
                )

            plt.title('物体检测结果统计', fontsize=14)
            plt.xlabel('数量', fontsize=12)
            plt.ylabel('物体类别', fontsize=12)
            plt.tight_layout()

            # 创建饼图
            plt.subplot(1, 2, 2)
            plt.pie(
                counts,
                labels=class_names,
                autopct='%1.1f%%',
                shadow=True,
                startangle=90
            )
            plt.axis('equal')
            plt.title('物体类别分布', fontsize=14)

            # 保存图表
            plt.tight_layout()
            plt.savefig(self.result_image)
            plt.close()

            logger.info(f"统计图表已保存: {self.result_image}")
        except Exception as e:
            logger.error(f"生成统计图表时出错: {str(e)}")
            self._generate_simple_statistics(results)

    def _generate_simple_statistics(self, results):
        """生成简化版的统计图表（作为备用）

        Args:
            results: 包含每个帧检测到的物体信息的列表
        """
        try:
            # 统计每个类别的数量
            class_counts = {}
            for result in results:
                class_name = result['class']
                if class_name in class_counts:
                    class_counts[class_name] += 1
                else:
                    class_counts[class_name] = 1

            # 按数量排序
            sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

            # 只取前10个最常见的类别
            if len(sorted_classes) > 10:
                sorted_classes = sorted_classes[:10]

            # 准备绘图数据
            class_names = [item[0] for item in sorted_classes]
            counts = [item[1] for item in sorted_classes]

            # 创建简单的条形图
            plt.figure(figsize=(10, 6))
            plt.bar(class_names, counts)
            plt.xticks(rotation=45, ha='right')
            plt.title('物体检测结果')
            plt.xlabel('物体类别')
            plt.ylabel('数量')
            plt.tight_layout()

            # 保存图表
            plt.savefig(self.result_image)
            plt.close()

            logger.info(f"简单统计图表已保存: {self.result_image}")
        except Exception as e:
            logger.error(f"生成简单统计图表也失败了: {str(e)}")
