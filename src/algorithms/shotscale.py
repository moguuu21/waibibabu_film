import os
import csv
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import logging
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import platform
import matplotlib.font_manager as fm

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ShotScale")

class ShotScale:
    def __init__(self, path):
        """初始化ShotScale类
        
        Args:
            path: 存放视频帧的目录路径
        """
        self.path = path
        self.frame_dir = os.path.join(path, "frame")
        self.frame_files = []
        self.shot_scales = []
        self.result_csv = os.path.join(path, "shotscale_results.csv")
        self.result_image = os.path.join(path, "shotscale.png")
        self.timeline_image = os.path.join(path, "shotscale_timeline.png")
        
        # 模型文件路径设置为本地目录
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        self.resnet_model_path = os.path.join(self.models_dir, "shotscale_resnet.pth")
        
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
        
        # 初始化模型
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {self.device}")
        
        # 设置中文字体
        self._setup_matplotlib_font()
        
        # 定义镜头尺度类别
        self.scale_categories = [
            "极远景 (Extreme Long Shot)",
            "远景 (Long Shot)",
            "全景 (Full Shot)",
            "中景 (Medium Shot)",
            "中近景 (Medium Close-up)",
            "特写 (Close-up)",
            "极特写 (Extreme Close-up)"
        ]
    
    def _setup_matplotlib_font(self):
        """配置matplotlib中文字体"""
        try:
            # 检测系统
            if platform.system() == "Windows":
                # Windows常用中文字体
                font_paths = [
                    "C:/Windows/Fonts/simhei.ttf",            # 黑体
                    "C:/Windows/Fonts/simsun.ttc",            # 宋体
                    "C:/Windows/Fonts/msyh.ttc",              # 微软雅黑
                    "C:/Windows/Fonts/Microsoft YaHei UI.ttc" # 微软雅黑UI
                ]
                
                # 检查字体是否存在并设置
                for font_path in font_paths:
                    if os.path.exists(font_path):
                        # 添加字体
                        font_prop = fm.FontProperties(fname=font_path)
                        # 设置默认字体
                        plt.rcParams['font.family'] = font_prop.get_name()
                        logger.info(f"已设置matplotlib中文字体: {font_path}")
                        break
                
            elif platform.system() == "Darwin":  # macOS
                # macOS常用中文字体
                plt.rcParams['font.family'] = ['Arial Unicode MS', 'STHeiti', 'SimHei']
                logger.info("已设置macOS中文字体")
                
            elif platform.system() == "Linux":
                # Linux常用中文字体
                plt.rcParams['font.family'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'SimHei']
                logger.info("已设置Linux中文字体")
            
            # 通用配置
            plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
            
        except Exception as e:
            logger.error(f"设置matplotlib中文字体失败: {str(e)}")
            # 尝试使用备用方法
            try:
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Bitstream Vera Sans']
                plt.rcParams['axes.unicode_minus'] = False
                logger.info("已使用备用方法设置中文字体")
            except Exception as e2:
                logger.error(f"备用中文字体设置也失败: {str(e2)}")
    
    def _load_model(self):
        """加载预训练模型
        
        Returns:
            model: 预训练的ResNet模型
        """
        try:
            # 检查模型文件是否存在
            if os.path.exists(self.resnet_model_path):
                # 加载本地预训练模型
                model = torch.load(self.resnet_model_path, map_location=self.device)
                logger.info(f"已加载本地预训练模型: {self.resnet_model_path}")
                return model
                
            # 如果本地模型不存在，加载预训练的ResNet50模型
            logger.info("本地模型文件不存在，使用OpenCV基于规则的方法")
            logger.info(f"请将预训练模型文件放置在: {self.resnet_model_path}")
            
            # 这里使用torchvision提供的预训练模型作为备选
            model = models.resnet50(pretrained=True)
            
            # 修改最后一层以适应镜头尺度分类
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, len(self.scale_categories))
            
            # 设置为评估模式
            model.eval()
            model = model.to(self.device)
            
            return model
        except Exception as e:
            logger.error(f"加载模型时出错: {str(e)}")
            return None
    
    def _preprocess_image(self, image_path):
        """预处理图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            img_tensor: 预处理后的图像张量
        """
        try:
            # 定义图像变换
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            # 打开并变换图像
            img = Image.open(image_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)  # 添加批处理维度
            
            return img_tensor.to(self.device)
        except Exception as e:
            logger.error(f"预处理图像时出错 {image_path}: {str(e)}")
            return None
    
    def _predict_scale(self, model, img_tensor):
        """预测镜头尺度
        
        Args:
            model: 预训练模型
            img_tensor: 预处理后的图像张量
            
        Returns:
            scale_category: 预测的镜头尺度类别
            confidence: 预测的置信度
        """
        try:
            if img_tensor is None:
                return None, 0.0
                
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                
                # 获取最可能的类别及其概率
                max_prob, predicted = torch.max(probabilities, 0)
                
                scale_category = self.scale_categories[predicted.item()]
                confidence = max_prob.item()
                
                return scale_category, confidence
        except Exception as e:
            logger.error(f"预测镜头尺度时出错: {str(e)}")
            return None, 0.0
    
    def _analyze_with_opencv(self, image_path):
        """使用OpenCV分析镜头尺度（基于启发式规则）
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            scale_category: 预测的镜头尺度类别
            confidence: 预测的置信度
        """
        try:
            # 读取图像
            image = cv2.imread(image_path)
            
            if image is None:
                logger.warning(f"无法读取图像: {image_path}")
                return None, 0.0
                
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 检测人脸
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # 获取图像尺寸
            height, width = gray.shape
            image_area = height * width
            
            # 基于启发式规则确定镜头尺度
            if len(faces) > 0:
                # 获取最大人脸区域
                max_face = max(faces, key=lambda x: x[2] * x[3])
                face_area = max_face[2] * max_face[3]
                face_ratio = face_area / image_area
                
                # 基于人脸占比确定镜头尺度
                if face_ratio > 0.2:
                    # 人脸占据画面大部分，可能是特写
                    return "特写 (Close-up)", 0.8
                elif face_ratio > 0.1:
                    # 人脸适中，可能是中近景
                    return "中近景 (Medium Close-up)", 0.7
                elif face_ratio > 0.05:
                    # 人脸较小，可能是中景
                    return "中景 (Medium Shot)", 0.6
                else:
                    # 人脸非常小，可能是全景或远景
                    return "全景 (Full Shot)", 0.5
            else:
                # 没有检测到人脸，进一步分析图像特征
                
                # 计算边缘密度
                edges = cv2.Canny(gray, 100, 200)
                edge_density = np.count_nonzero(edges) / image_area
                
                # 计算纹理特征
                texture_features = cv2.calcHist([gray], [0], None, [16], [0, 256])
                texture_variance = np.var(texture_features)
                
                # 基于边缘密度和纹理特征确定镜头尺度
                if edge_density < 0.01 and texture_variance < 100:
                    # 边缘少，纹理变化小，可能是极远景
                    return "极远景 (Extreme Long Shot)", 0.4
                elif edge_density < 0.03:
                    # 边缘较少，可能是远景
                    return "远景 (Long Shot)", 0.5
                else:
                    # 默认为全景
                    return "全景 (Full Shot)", 0.3
        except Exception as e:
            logger.error(f"使用OpenCV分析镜头尺度时出错 {image_path}: {str(e)}")
            # 默认为中景
            return "中景 (Medium Shot)", 0.2
    
    def shotscale_recognize(self):
        """识别所有帧的镜头尺度
        
        Returns:
            results: 包含帧ID和镜头尺度分析结果的列表
        """
        try:
            logger.info("开始镜头尺度分析...")
            
            # 检查frame目录是否存在
            if not os.path.exists(self.frame_dir):
                logger.error(f"帧图像目录不存在: {self.frame_dir}")
                return []
            
            # 获取所有帧图像
            self.frame_files = sorted([
                f for f in os.listdir(self.frame_dir) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            
            if not self.frame_files:
                logger.warning(f"在 {self.frame_dir} 中未找到图像文件")
                return []
            
            logger.info(f"共有 {len(self.frame_files)} 个图像文件需要分析")
            
            # 尝试加载模型
            model = self._load_model()
            use_model = model is not None
            
            if use_model:
                logger.info("使用预训练模型进行镜头尺度识别")
            else:
                logger.info("使用OpenCV基于规则的方法进行镜头尺度识别")
            
            # 开始处理图像
            results = []
            
            for i, frame_file in enumerate(self.frame_files):
                if i % 10 == 0:
                    logger.info(f"正在处理第 {i+1}/{len(self.frame_files)} 个图像")
                
                try:
                    frame_id = frame_file.replace("frame", "").replace(".jpg", "").replace(".png", "")
                    image_path = os.path.join(self.frame_dir, frame_file)
                    
                    if use_model:
                        # 使用预训练模型
                        img_tensor = self._preprocess_image(image_path)
                        scale_category, confidence = self._predict_scale(model, img_tensor)
                    else:
                        # 使用OpenCV基于规则的方法
                        scale_category, confidence = self._analyze_with_opencv(image_path)
                    
                    if scale_category:
                        results.append({
                            'frame_id': frame_id,
                            'scale_category': scale_category,
                            'confidence': confidence
                        })
                except Exception as e:
                    logger.error(f"处理图像 {frame_file} 时出错: {str(e)}")
            
            logger.info(f"镜头尺度分析完成，共分析 {len(results)} 个图像")
            
            # 保存结果到CSV
            with open(self.result_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['frame_id', 'scale_category', 'confidence'])
                
                for result in results:
                    writer.writerow([
                        result['frame_id'],
                        result['scale_category'],
                        result['confidence']
                    ])
            
            # 生成统计图表
            self._generate_statistics(results)
            
            return results
        except Exception as e:
            logger.error(f"分析镜头尺度时出错: {str(e)}")
            return []
    
    def _generate_statistics(self, results):
        """生成统计图表
        
        Args:
            results: 分析结果列表
        """
        if not results:
            logger.warning("没有可用的分析结果来生成统计图表")
            return
        
        try:
            # 确保中文字体已设置
            self._setup_matplotlib_font()
            
            # 按镜头尺度类型分组统计数量
            scale_counts = {}
            for result in results:
                scale_type = result['scale_category']
                if scale_type in scale_counts:
                    scale_counts[scale_type] += 1
                else:
                    scale_counts[scale_type] = 1
            
            # 准备图表数据
            scale_types = list(scale_counts.keys())
            counts = list(scale_counts.values())
            
            # 按类型的常规顺序排序
            scale_order = [
                "特写 (Close-up)", 
                "中近景 (Medium Close-up)", 
                "中景 (Medium Shot)",
                "全景 (Full Shot)",
                "远景 (Long Shot)",
                "极远景 (Extreme Long Shot)"
            ]
            
            # 获取所有存在的尺度类型并按常规顺序排序
            ordered_data = []
            for scale in scale_order:
                if scale in scale_counts:
                    ordered_data.append((scale, scale_counts[scale]))
            
            # 分离排序后的数据
            if ordered_data:
                scale_types, counts = zip(*ordered_data)
            
            # 创建颜色映射
            colors = ['#FF6B6B', '#4ECDC4', '#1A535C', '#3D5A80', '#E9C46A', '#2A9D8F']
            color_dict = {scale: color for scale, color in zip(scale_order, colors)}
            
            # 创建饼图
            plt.figure(figsize=(12, 8))
            plt.subplot(1, 2, 1)
            wedges, texts, autotexts = plt.pie(
                counts, 
                labels=scale_types,
                autopct='%1.1f%%',
                startangle=90,
                colors=[color_dict.get(scale, '#999999') for scale in scale_types]
            )
            
            # 设置饼图标题和属性
            plt.title('镜头尺度分布', fontsize=14)
            plt.axis('equal')
            
            # 为标签和百分比文本设置样式
            for text in texts:
                text.set_fontsize(10)
            for autotext in autotexts:
                autotext.set_fontsize(10)
                autotext.set_color('white')
            
            # 创建条形图
            plt.subplot(1, 2, 2)
            bars = plt.barh(
                scale_types, 
                counts,
                color=[color_dict.get(scale, '#999999') for scale in scale_types]
            )
            
            # 在条形上显示数值
            for bar, count in zip(bars, counts):
                plt.text(
                    bar.get_width() + 0.5, 
                    bar.get_y() + bar.get_height()/2,
                    f'{count} ({count/sum(counts)*100:.1f}%)',
                    va='center'
                )
            
            # 设置条形图标题和标签
            plt.title('镜头尺度数量统计', fontsize=14)
            plt.xlabel('数量', fontsize=12)
            plt.tight_layout()
            
            # 保存图表
            plt.savefig(self.result_image, bbox_inches="tight")
            plt.close()
            
            logger.info(f"镜头尺度统计图表已保存: {self.result_image}")
            
            # 生成时间线图表
            self._generate_timeline_chart(results)
        except Exception as e:
            logger.error(f"生成统计图表失败: {str(e)}")
            # 尝试生成简单的统计图表
            self._generate_simple_statistics(results)
    
    def _generate_timeline_chart(self, results):
        """生成随时间变化的镜头尺度图表
        
        Args:
            results: 包含帧ID和镜头尺度分析结果的列表
        """
        try:
            # 确保中文字体已设置
            self._setup_matplotlib_font()
            
            # 按帧ID排序结果
            sorted_results = sorted(results, key=lambda x: int(x['frame_id']))
            
            # 提取数据
            frame_ids = [int(result['frame_id']) for result in sorted_results]
            scales = [result['scale_category'] for result in sorted_results]
            confidences = [result['confidence'] for result in sorted_results]
            
            # 为每个尺度类型分配唯一的数值表示
            scale_types = list(set(scales))
            scale_values = {scale: i for i, scale in enumerate(sorted(scale_types))}
            
            # 将尺度转换为数值
            numeric_scales = [scale_values[scale] for scale in scales]
            
            # 创建图表
            plt.figure(figsize=(14, 8))
            
            # 绘制尺度变化曲线，用点的大小表示置信度
            scatter = plt.scatter(
                frame_ids, 
                numeric_scales,
                c=numeric_scales,
                cmap='viridis',
                s=[conf * 100 for conf in confidences],  # 置信度越高，点越大
                alpha=0.7
            )
            
            # 设置Y轴刻度为尺度名称
            plt.yticks(
                range(len(scale_types)),
                sorted(scale_types)
            )
            
            # 设置标题和轴标签
            plt.title('镜头尺度随时间变化', fontsize=16)
            plt.xlabel('帧ID', fontsize=14)
            plt.ylabel('镜头尺度', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 添加颜色图例
            cbar = plt.colorbar(scatter)
            cbar.set_label('镜头尺度类型')
            
            # 添加置信度图例
            conf_levels = [0.2, 0.5, 0.8]
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor='grey', 
                          markersize=np.sqrt(conf * 100 / np.pi), 
                          label=f'置信度: {conf}')
                for conf in conf_levels
            ]
            plt.legend(handles=legend_elements, loc='upper right')
            
            plt.tight_layout()
            
            # 保存图表
            plt.savefig(self.timeline_image, dpi=100, bbox_inches="tight")
            plt.close()
            
            logger.info(f"镜头尺度时间线图表已保存: {self.timeline_image}")
        except Exception as e:
            logger.error(f"生成镜头尺度时间线图表失败: {str(e)}")
    
    def _generate_simple_statistics(self, results):
        """生成简单的镜头尺度统计图表（备选方案）
        
        Args:
            results: 包含帧ID和镜头尺度分析结果的列表
        """
        try:
            # 统计各类镜头尺度的数量
            scale_counts = {}
            for result in results:
                scale_type = result['scale_category']
                if scale_type in scale_counts:
                    scale_counts[scale_type] += 1
                else:
                    scale_counts[scale_type] = 1
                    
            # 创建简单条形图
            plt.figure(figsize=(10, 6))
            plt.bar(scale_counts.keys(), scale_counts.values())
            plt.title('镜头尺度统计')
            plt.xlabel('镜头尺度类型')
            plt.ylabel('数量')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # 保存图表
            plt.savefig(self.result_image, bbox_inches="tight")
            plt.close()
            
            logger.info(f"简单镜头尺度统计图表已保存: {self.result_image}")
        except Exception as e:
            logger.error(f"生成简单统计图表失败: {str(e)}")