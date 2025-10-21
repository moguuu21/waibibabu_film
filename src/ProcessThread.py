import os
import time
import logging
from PySide6.QtCore import QThread, Signal
from algorithms.shotcutTransNetV2 import TransNetV2
from algorithms.objectDetection import ObjectDetection
from algorithms.colorAnalyzer import ColorAnalyzer
# 移除Tesseract版本的字幕处理
# from algorithms.subtitles import Subtitles
from algorithms.subtitleEasyOcr import SubtitleProcessor
from algorithms.shotscale import ShotScale

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ProcessThread")

class ProcessThread(QThread):
    progress_signal = Signal(int, str)  # 进度信号，参数为进度百分比和描述
    finish_signal = Signal(bool, str)   # 完成信号，参数为是否成功和结果描述
    
    def __init__(self, task_type, input_path=None, parent=None, **kwargs):
        """初始化处理线程
        
        Args:
            task_type: 任务类型，如"shotcut", "color", "object", "subtitle", "shotscale"
            input_path: 输入路径，通常是保存图像帧的目录的父目录
            parent: 父对象
            **kwargs: 额外的参数，包括：
                v_path: 视频文件路径
                image_save: 图像保存路径 
                frame_save: 帧保存路径
                th: 阈值
                imgpath: 图像路径
                colors_count: 颜色数量
                filename: 文件名
                subtitle_value: 字幕阈值
        """
        super(ProcessThread, self).__init__(parent)
        self.task_type = task_type
        self.input_path = input_path
        self.parent = parent
        self.is_running = True
        
        # 保存额外的参数
        self.kwargs = kwargs
        self.v_path = kwargs.get('v_path')
        self.image_save = kwargs.get('image_save')
        self.frame_save = kwargs.get('frame_save')
        self.th = kwargs.get('th')
        self.imgpath = kwargs.get('imgpath')
        self.colors_count = kwargs.get('colors_count')
        self.filename = kwargs.get('filename')
        self.subtitle_value = kwargs.get('subtitle_value')
    
    def run(self):
        """线程主函数，根据任务类型执行不同的处理"""
        try:
            logger.info(f"开始执行任务: {self.task_type}, 输入路径: {self.input_path}")
            
            # 确保有有效的输入路径
            if self.task_type == "shotcut" and self.v_path:
                # 对于shotcut任务，使用v_path参数
                if not os.path.exists(self.v_path):
                    logger.error(f"视频文件不存在: {self.v_path}")
                    self.finish_signal.emit(False, f"视频文件不存在: {self.v_path}")
                    return
            elif self.input_path:
                # 对于其他任务，检查input_path
                if not os.path.exists(self.input_path):
                    logger.error(f"输入路径不存在: {self.input_path}")
                    self.finish_signal.emit(False, f"输入路径不存在: {self.input_path}")
                    return
            else:
                # 如果两者都不存在
                logger.error("无效的输入路径")
                self.finish_signal.emit(False, "请提供有效的输入路径")
                return
            
            # 根据任务类型执行不同的处理
            if self.task_type == "shotcut":
                self._process_shotcut()
            elif self.task_type == "color":
                self._process_color()
            elif self.task_type == "object":
                self._process_object()
            elif self.task_type == "subtitle":
                self._process_subtitle()
            elif self.task_type == "shotscale":
                self._process_shotscale()
            else:
                logger.error(f"未知的任务类型: {self.task_type}")
                self.finish_signal.emit(False, f"未知的任务类型: {self.task_type}")
        except Exception as e:
            logger.error(f"处理任务时出错: {str(e)}")
            self.finish_signal.emit(False, f"处理失败: {str(e)}")
    
    def stop(self):
        """停止线程"""
        self.is_running = False
        self.wait()
    
    def _process_shotcut(self):
        """处理镜头切分任务"""
        try:
            self.progress_signal.emit(10, "正在初始化镜头切分...")
            
            # 检查必要参数
            if not self.v_path or not os.path.exists(self.v_path):
                self.finish_signal.emit(False, f"视频文件不存在: {self.v_path}")
                return
                
            if not self.image_save:
                self.image_save = os.path.dirname(self.v_path)
                
            # 构建帧保存路径
            frame_save = os.path.join(self.image_save, "frame")
            os.makedirs(frame_save, exist_ok=True)
            
            # 初始化镜头切分模型 - 不传递input_path，使用默认模型目录
            shotcut = TransNetV2()
            
            self.progress_signal.emit(20, "正在分析视频...")
            
            # 执行镜头切分，传递正确的参数
            result = shotcut.shotcut_detection(
                v_path=self.v_path,
                image_save=self.image_save,
                frame_save=frame_save,
                th=self.th if self.th else 0.5
            )
            
            self.progress_signal.emit(80, "正在生成结果...")
            
            if result and len(result) > 0:
                logger.info(f"镜头切分完成，检测到 {len(result)} 个镜头")
                self.finish_signal.emit(True, f"镜头切分完成，检测到 {len(result)} 个镜头")
            else:
                logger.warning("镜头切分未检测到任何镜头")
                self.finish_signal.emit(False, "镜头切分未检测到任何镜头")
        except Exception as e:
            logger.error(f"镜头切分处理时出错: {str(e)}")
            self.finish_signal.emit(False, f"镜头切分失败: {str(e)}")
    
    def _process_color(self):
        """处理色彩分析任务"""
        try:
            self.progress_signal.emit(10, "正在初始化色彩分析...")
            
            # 检查输入路径是否存在
            if not os.path.exists(self.input_path):
                logger.error(f"输入路径不存在: {self.input_path}")
                self.finish_signal.emit(False, f"输入路径不存在: {self.input_path}")
                return
            
            # 确保frame目录存在
            frame_dir = os.path.join(self.input_path, "frame")
            if not os.path.exists(frame_dir):
                os.makedirs(frame_dir, exist_ok=True)
                logger.info(f"已创建frame目录: {frame_dir}")
                
                # 如果frame目录是空的，但有视频文件，则从视频中提取帧
                if self.v_path and os.path.exists(self.v_path):
                    self.progress_signal.emit(20, "正在从视频中提取帧...")
                    # 调用TransNetV2提取帧
                    transnet = TransNetV2()
                    transnet.getFrame_number(self.v_path, self.input_path)
                    self.progress_signal.emit(50, "帧提取完成，开始分析色彩...")
            
            # 初始化色彩分析
            colors_count = self.kwargs.get('colors_count', 5)  # 默认5种颜色
            
            self.progress_signal.emit(60, f"正在分析色彩，提取{colors_count}种主要颜色...")
            
            # 执行色彩分析
            color_analyzer = ColorAnalyzer(self.input_path)
            result = color_analyzer.analyze_colors(colors_count)
            
            self.progress_signal.emit(90, "色彩分析完成!")
            
            self.progress_signal.emit(100, "色彩分析完成!")
            self.finish_signal.emit(True, "色彩分析完成!")
        except Exception as e:
            logger.error(f"色彩分析处理时出错: {str(e)}")
            self.finish_signal.emit(False, f"色彩分析失败: {str(e)}")
    
    def _process_object(self):
        """处理物体检测任务"""
        try:
            self.progress_signal.emit(10, "正在初始化物体检测...")
            
            # 检查imgpath参数，构建输入路径
            if self.imgpath:
                img_dir = f"./img/{self.imgpath}"
                if not os.path.exists(img_dir):
                    os.makedirs(img_dir, exist_ok=True)
                    
                # 确保frame目录存在
                frame_dir = os.path.join(img_dir, "frame")
                if not os.path.exists(frame_dir):
                    os.makedirs(frame_dir, exist_ok=True)
                    
                self.input_path = img_dir
            
            # 初始化物体检测
            object_detector = ObjectDetection(self.input_path)
            
            self.progress_signal.emit(20, "正在检测物体...")
            
            # 执行物体检测
            result = object_detector.object_detection()
            
            self.progress_signal.emit(80, "正在生成结果...")
            
            if result:
                logger.info(f"物体检测完成，检测到 {len(result)} 个物体")
                self.finish_signal.emit(True, f"物体检测完成，检测到 {len(result)} 个物体")
            else:
                logger.warning("物体检测未检测到任何物体")
                self.finish_signal.emit(False, "物体检测未检测到任何物体")
        except Exception as e:
            logger.error(f"物体检测处理时出错: {str(e)}")
            self.finish_signal.emit(False, f"物体检测失败: {str(e)}")
    
    def _process_subtitle(self):
        """处理字幕识别任务"""
        try:
            if not os.path.exists(self.input_path):
                raise FileNotFoundError(f"视频文件不存在: {self.input_path}")

            # 从Tesseract OCR切换到EasyOCR
            self.subtitle_processor = SubtitleProcessor()
            subtitle_value = self.kwargs.get('subtitle_value', 48)  # 默认值48

            self.progress_signal.emit(10, "正在识别字幕...")
            
            # 获取视频文件名，用于保存结果
            video_name = os.path.basename(self.input_path).split('.')[0]
            save_path = f"./img/{video_name}/"
            
            # 确保保存目录存在
            os.makedirs(save_path, exist_ok=True)
            
            # 执行字幕识别
            subtitle_str, subtitle_list = self.subtitle_processor.getsubtitleEasyOcr(
                self.input_path, save_path, subtitle_value
            )
            
            self.progress_signal.emit(80, "正在保存字幕结果...")
            
            # 保存为SRT文件
            self.subtitle_processor.subtitle2Srt(subtitle_list, save_path)
            
            self.progress_signal.emit(100, "字幕识别完成!")
            self.finish_signal.emit(True, "字幕识别完成!")
        except Exception as e:
            logger.error(f"字幕识别失败: {str(e)}")
            self.finish_signal.emit(False, f"字幕识别失败: {str(e)}")
    
    def _process_shotscale(self):
        """处理镜头尺度分析任务"""
        try:
            self.progress_signal.emit(10, "正在初始化镜头尺度分析...")
            
            # 检查输入路径是否存在
            if not os.path.exists(self.input_path):
                logger.error(f"输入路径不存在: {self.input_path}")
                self.finish_signal.emit(False, f"输入路径不存在: {self.input_path}")
                return
            
            # 确保frame目录存在
            frame_dir = os.path.join(self.input_path, "frame")
            if not os.path.exists(frame_dir):
                os.makedirs(frame_dir, exist_ok=True)
                logger.info(f"已创建frame目录: {frame_dir}")
                
                # 如果frame目录是空的，但有视频文件，则从视频中提取帧
                if self.v_path and os.path.exists(self.v_path):
                    self.progress_signal.emit(20, "正在从视频中提取帧...")
                    # 调用TransNetV2提取帧
                    transnet = TransNetV2()
                    transnet.getFrame_number(self.v_path, self.input_path)
                    self.progress_signal.emit(50, "帧提取完成，开始分析镜头尺度...")
            
            # 初始化镜头尺度分析
            shotscale = ShotScale(self.input_path)
            
            self.progress_signal.emit(60, "正在分析镜头尺度...")
            
            # 执行镜头尺度分析
            result = shotscale.shotscale_recognize()
            
            self.progress_signal.emit(100, "镜头尺度分析完成!")
            if result:
                self.finish_signal.emit(True, f"镜头尺度分析完成，分析了 {len(result)} 个镜头")
            else:
                self.finish_signal.emit(False, "镜头尺度分析未检测到任何镜头")
        except Exception as e:
            logger.error(f"镜头尺度分析处理时出错: {str(e)}")
            self.finish_signal.emit(False, f"镜头尺度分析失败: {str(e)}") 