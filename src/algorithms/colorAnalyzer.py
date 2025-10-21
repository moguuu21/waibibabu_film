import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import logging
from sklearn.cluster import KMeans
from matplotlib.colors import rgb2hex
import math
import platform
import matplotlib.font_manager as fm

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ColorAnalyzer")

class ColorAnalyzer:
    def __init__(self, path, n_clusters=5):
        """初始化色彩分析类
        
        Args:
            path: 图像保存路径
            n_clusters: 颜色聚类数量
        """
        self.path = path
        self.frame_folder = os.path.join(path, "frame")
        self.result_csv = os.path.join(path, "color.csv")
        self.result_image = os.path.join(path, "color.png")
        self.n_clusters = n_clusters
        
        # 检查图像文件夹是否存在
        if not os.path.exists(self.frame_folder):
            os.makedirs(self.frame_folder)
            logger.warning(f"图像文件夹不存在，已创建: {self.frame_folder}")
        
        # 设置中文字体
        self._setup_matplotlib_font()
    
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
    
    def _extract_colors(self, image):
        """从图像中提取主要颜色
        
        Args:
            image: 输入图像
            
        Returns:
            colors: 主要颜色列表 [(hex_color, percentage), ...]
        """
        try:
            # 调整图像大小以加快处理速度
            image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA)
            
            # 重塑图像为二维数组（每行一个像素，每列一个通道）
            pixels = image.reshape(-1, 3)
            
            # 将BGR转换为RGB（OpenCV默认是BGR）
            pixels = pixels[:, ::-1]
            
            # 使用K-means聚类
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
            kmeans.fit(pixels)
            
            # 获取聚类中心（主要颜色）
            colors = kmeans.cluster_centers_
            
            # 计算每种颜色的百分比
            labels = kmeans.labels_
            counts = np.bincount(labels)
            percentages = counts / len(labels)
            
            # 将颜色转换为十六进制格式并添加百分比
            hex_colors = []
            for i, color in enumerate(colors):
                hex_color = rgb2hex(np.array(color/255))
                hex_colors.append((hex_color, percentages[i]))
            
            # 按百分比排序
            hex_colors = sorted(hex_colors, key=lambda x: x[1], reverse=True)
            
            return hex_colors
        except Exception as e:
            logger.error(f"提取颜色时出错: {str(e)}")
            return []
    
    def _calculate_brightness(self, image):
        """计算图像亮度
        
        Args:
            image: 输入图像
            
        Returns:
            brightness: 亮度值（0-255）
        """
        try:
            # 转换为HSV颜色空间
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 提取V通道（亮度）并计算平均值
            brightness = np.mean(hsv[:, :, 2])
            
            return brightness
        except Exception as e:
            logger.error(f"计算亮度时出错: {str(e)}")
            return 0
    
    def _calculate_contrast(self, image):
        """计算图像对比度
        
        Args:
            image: 输入图像
            
        Returns:
            contrast: 对比度值
        """
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 计算标准差作为对比度指标
            contrast = np.std(gray.flatten())
            
            return contrast
        except Exception as e:
            logger.error(f"计算对比度时出错: {str(e)}")
            return 0
    
    def _calculate_saturation(self, image):
        """计算图像饱和度
        
        Args:
            image: 输入图像
            
        Returns:
            saturation: 饱和度值（0-1）
        """
        try:
            # 转换为HSV颜色空间
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 提取S通道（饱和度）并计算平均值
            saturation = np.mean(hsv[:, :, 1]) / 255.0
            
            return saturation
        except Exception as e:
            logger.error(f"计算饱和度时出错: {str(e)}")
            return 0
    
    def _calculate_color_variance(self, image):
        """计算图像颜色方差
        
        Args:
            image: 输入图像
            
        Returns:
            variance: 颜色方差值
        """
        try:
            # 分离BGR通道
            b, g, r = cv2.split(image)
            
            # 计算每个通道的标准差
            b_std = np.std(b)
            g_std = np.std(g)
            r_std = np.std(r)
            
            # 计算合并方差
            variance = math.sqrt(b_std**2 + g_std**2 + r_std**2)
            
            return variance
        except Exception as e:
            logger.error(f"计算颜色方差时出错: {str(e)}")
            return 0
    
    def analyze_colors(self, colors_count=None):
        """分析视频帧中的颜色
        
        Args:
            colors_count: 颜色聚类数量，如果提供则更新self.n_clusters
            
        Returns:
            results: 包含帧ID和颜色分析结果的列表
        """
        try:
            # 如果提供了colors_count参数，更新n_clusters
            if colors_count is not None:
                self.n_clusters = int(colors_count)
                logger.info(f"更新颜色聚类数量为: {self.n_clusters}")
            
            # 检查图像文件夹是否存在
            if not os.path.exists(self.frame_folder):
                logger.error(f"图像文件夹不存在: {self.frame_folder}")
                return []
            
            # 获取所有图像文件
            image_files = [f for f in os.listdir(self.frame_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            if not image_files:
                logger.warning(f"没有找到图像文件: {self.frame_folder}")
                return []
            
            # 分析每一帧图像
            results = []
            
            logger.info(f"开始分析颜色，共{len(image_files)}张图像")
            
            for i, image_file in enumerate(sorted(image_files)):
                if i % 10 == 0:
                    logger.info(f"正在处理第{i+1}/{len(image_files)}张图像...")
                
                try:
                    # 提取帧ID
                    frame_id = image_file.replace("frame", "").replace(".jpg", "").replace(".jpeg", "").replace(".png", "")
                    
                    # 读取图像
                    image_path = os.path.join(self.frame_folder, image_file)
                    image = cv2.imread(image_path)
                    
                    if image is None:
                        logger.warning(f"无法读取图像: {image_path}")
                        continue
                    
                    # 提取主要颜色
                    colors = self._extract_colors(image)
                    
                    # 计算图像特征
                    brightness = self._calculate_brightness(image)
                    contrast = self._calculate_contrast(image)
                    saturation = self._calculate_saturation(image)
                    color_variance = self._calculate_color_variance(image)
                    
                    # 收集结果
                    result = {
                        'frame_id': frame_id,
                        'colors': colors,
                        'brightness': brightness,
                        'contrast': contrast,
                        'saturation': saturation,
                        'color_variance': color_variance
                    }
                    
                    results.append(result)
                except Exception as e:
                    logger.error(f"处理图像{image_file}时出错: {str(e)}")
            
            # 保存结果到CSV文件
            logger.info(f"共分析了{len(results)}个图像，保存结果...")
            
            with open(self.result_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['frame_id', 'primary_color', 'brightness', 'contrast', 'saturation', 'color_variance'])
                
                for result in results:
                    primary_color = result['colors'][0][0] if result['colors'] else '#000000'
                    writer.writerow([
                        result['frame_id'],
                        primary_color,
                        result['brightness'],
                        result['contrast'],
                        result['saturation'],
                        result['color_variance']
                    ])
            
            # 生成统计图表
            self._generate_statistics(results)
            
            return results
        except Exception as e:
            logger.error(f"分析颜色时出错: {str(e)}")
            return []
    
    def _generate_statistics(self, results):
        """生成色彩分析统计图表
        
        Args:
            results: 包含帧ID和色彩分析结果的列表
        """
        try:
            # 确保中文字体已设置
            self._setup_matplotlib_font()
            
            # 按帧ID排序结果
            sorted_results = sorted(results, key=lambda x: int(x['frame_id']))
            
            # 提取数据
            frame_ids = [int(result['frame_id']) for result in sorted_results]
            brightness_values = [result['brightness'] for result in sorted_results]
            contrast_values = [result['contrast'] for result in sorted_results]
            saturation_values = [result['saturation'] for result in sorted_results]
            
            # 创建图表
            fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
            
            # 亮度图表
            axs[0].plot(frame_ids, brightness_values, color='#1976D2', linewidth=2)
            axs[0].fill_between(frame_ids, brightness_values, color='#1976D2', alpha=0.2)
            axs[0].set_title('亮度变化', fontsize=14)
            axs[0].set_ylabel('亮度 (0-255)', fontsize=12)
            axs[0].grid(True, linestyle='--', alpha=0.7)
            
            # 对比度图表
            axs[1].plot(frame_ids, contrast_values, color='#D81B60', linewidth=2)
            axs[1].fill_between(frame_ids, contrast_values, color='#D81B60', alpha=0.2)
            axs[1].set_title('对比度变化', fontsize=14)
            axs[1].set_ylabel('对比度', fontsize=12)
            axs[1].grid(True, linestyle='--', alpha=0.7)
            
            # 饱和度图表
            axs[2].plot(frame_ids, saturation_values, color='#2E7D32', linewidth=2)
            axs[2].fill_between(frame_ids, saturation_values, color='#2E7D32', alpha=0.2)
            axs[2].set_title('饱和度变化', fontsize=14)
            axs[2].set_xlabel('帧ID', fontsize=12)
            axs[2].set_ylabel('饱和度', fontsize=12)
            axs[2].grid(True, linestyle='--', alpha=0.7)
            
            # 设置X轴刻度
            # 如果帧数太多，只显示一部分刻度
            if len(frame_ids) > 20:
                step = len(frame_ids) // 20
                axs[2].set_xticks(frame_ids[::step])
            else:
                axs[2].set_xticks(frame_ids)
            
            plt.tight_layout()
            plt.savefig(self.result_image, bbox_inches="tight")
            plt.close()
            
            logger.info(f"色彩分析统计图表已保存: {self.result_image}")
            
            # 生成调色板图表
            self._generate_palette_chart(results)
        except Exception as e:
            logger.error(f"生成统计图表失败: {str(e)}")
            # 尝试生成简单的统计图表
            self._generate_simple_statistics(results)
    
    def _generate_palette_chart(self, results):
        """生成视频的颜色调色板图表
        
        Args:
            results: 包含帧ID和颜色分析结果的列表
        """
        try:
            # 提取每个关键帧的主要颜色
            frame_count = len(results)
            
            if frame_count == 0:
                return
            
            # 选择最多30个关键帧
            step = max(1, frame_count // 30)
            key_frames = results[::step]
            
            # 创建图表
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 生成颜色条
            for i, result in enumerate(key_frames):
                colors = result['colors']
                frame_id = result['frame_id']
                
                # 确保有颜色数据
                if not colors:
                    continue
                
                # 绘制颜色条
                for j, (color, percentage) in enumerate(colors):
                    x_start = i
                    x_end = i + 1
                    y_start = j * (1.0 / self.n_clusters)
                    y_end = (j + 1) * (1.0 / self.n_clusters)
                    
                    rect = plt.Rectangle(
                        (x_start, y_start), x_end - x_start, y_end - y_start,
                        color=color, alpha=0.8
                    )
                    ax.add_patch(rect)
                    
                    # 对于主要颜色，添加帧ID标签
                    if j == 0 and i % 3 == 0:
                        ax.text(
                            x_start + 0.5, y_start - 0.05, frame_id,
                            horizontalalignment='center',
                            verticalalignment='bottom',
                            fontsize=8
                        )
            
            # 设置图表属性
            ax.set_xlim(0, len(key_frames))
            ax.set_ylim(0, 1)
            ax.set_title('视频颜色调色板', fontsize=14)
            ax.set_xlabel('帧序列', fontsize=12)
            ax.set_yticks([])
            ax.set_xticks([])
            
            # 保存图表
            palette_path = os.path.join(self.path, "color_palette.png")
            plt.savefig(palette_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            logger.info(f"颜色调色板已保存: {palette_path}")
        except Exception as e:
            logger.error(f"生成颜色调色板失败: {str(e)}")
    
    def _generate_simple_statistics(self, results):
        """生成简单的颜色统计图表（备选方案）
        
        Args:
            results: 包含帧ID和颜色分析结果的列表
        """
        try:
            # 按时间顺序排列结果
            sorted_results = sorted(results, key=lambda x: int(x['frame_id']))
            
            # 提取数据
            frame_ids = [int(result['frame_id']) for result in sorted_results]
            brightness_values = [result['brightness'] for result in sorted_results]
            
            # 创建简单图表
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(frame_ids, brightness_values)
            ax.set_title('亮度变化')
            ax.set_xlabel('帧ID')
            ax.set_ylabel('亮度')
            
            # 保存图表
            plt.savefig(self.result_image, bbox_inches="tight")
            plt.close()
            
            logger.info(f"简单统计图表已保存: {self.result_image}")
        except Exception as e:
            logger.error(f"生成简单统计图表也失败了: {str(e)}")
    
    def color_analyze(self):
        """执行颜色分析
        
        Returns:
            results: 包含帧ID和颜色分析结果的列表
        """
        logger.info("开始执行颜色分析...")
        return self.analyze_colors() 