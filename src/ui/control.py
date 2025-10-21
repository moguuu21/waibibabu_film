import os
import functools
import subprocess
import sys
import platform
import cv2
import numpy as np
import time

from PySide6.QtWidgets import (
    QDockWidget, QPushButton, QLabel, QFileDialog, QSlider, QMessageBox, QVBoxLayout,
    QWidget, QGridLayout, QHBoxLayout, QComboBox, QSpinBox, QLineEdit, QGroupBox,
    QScrollArea, QProgressDialog, QMenu
)
from PySide6.QtGui import QPixmap, QIcon, QImage
from PySide6.QtCore import Qt, QSize, Signal, QPoint
from algorithms.objectDetection import ObjectDetection
from algorithms.resultsave import resultsave
from algorithms.shotscale import ShotScale
from algorithms.shotcutTransNetV2 import TransNetV2
from algorithms.colorAnalyzer import ColorAnalyzer
from algorithms.subtitleEasyOcr import SubtitleProcessor

class Control(QDockWidget):
    sgn_shotcut = Signal(float)  # 分镜头检测信号
    sgn_colors = Signal(int)  # 色彩分析信号
    sgn_objects = Signal()  # 物体检测信号
    sgn_subtitle = Signal(float)  # 字幕识别信号
    sgn_shotscale = Signal()  # 镜头尺度信号

    def __init__(self, parent, filename=None):
        super().__init__('控制面板', parent)
        self.parent = parent
        self.filename = filename
        self.AnalyzeImg = ""
        self.AnalyzeImgPath = ""
        self.parent.filename_changed.connect(self.on_filename_changed)
        # self.video_info_loaded.connect(self.update_table)
        self.th = 0.35
        self.colors_count = 5  # 用于存储颜色数量的变量
        self.subtitleValue = 48
        self.init_ui()

    def init_ui(self):
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        # 创建主Widget
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # 允许内容小部件根据需要调整大小
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 禁用水平滚动条
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # 根据需要显示垂直滚动条
        
        # 设置滚动区域的样式表，使滚动条更美观
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background: #F0F0F0;
                width: 10px;
                margin: 0px 0px 0px 0px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #BBB;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #1976D2;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """)
        
        # 创建内容widget
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(10, 10, 10, 10)  # 设置边距
        content_layout.setSpacing(10)  # 设置空间
        
        # 分镜头检测按钮组
        shotcut_group = QGroupBox("分镜头检测")
        shotcut_layout = QVBoxLayout()
        shotcut_layout.setContentsMargins(10, 10, 10, 10)
        shotcut_label = QLabel("检测阈值")
        self.shotcutSlider = QSlider(Qt.Horizontal)
        self.shotcutSlider.setRange(400, 900)
        self.shotcutSlider.setValue(600)
        self.shotcutSlider.setTickPosition(QSlider.TicksBelow)
        self.shotcutSlider.setTickInterval(100)
        self.btnShotcut = QPushButton("分镜头检测")
        self.btnShotcut.setMinimumHeight(40)  # 设置最小高度
        self.btnShotcut.clicked.connect(self.on_shotcut_detection_button_clicked)
        
        shotcut_layout.addWidget(shotcut_label)
        shotcut_layout.addWidget(self.shotcutSlider)
        shotcut_layout.addWidget(self.btnShotcut)
        shotcut_group.setLayout(shotcut_layout)
        content_layout.addWidget(shotcut_group)
        
        # 色彩分析按钮组
        colors_group = QGroupBox("色彩分析")
        colors_layout = QVBoxLayout()
        colors_layout.setContentsMargins(10, 10, 10, 10)
        colors_label = QLabel("颜色数量")
        self.colorsC = QSlider(Qt.Horizontal)
        self.colorsC.setRange(3, 10)
        self.colorsC.setValue(5)
        self.colorsC.setTickPosition(QSlider.TicksBelow)
        self.colorsC.setTickInterval(1)
        self.btnColors = QPushButton("色彩分析")
        self.btnColors.setMinimumHeight(40)  # 设置最小高度
        self.btnColors.clicked.connect(self.on_colors_button_clicked)
        
        colors_layout.addWidget(colors_label)
        colors_layout.addWidget(self.colorsC)
        colors_layout.addWidget(self.btnColors)
        colors_group.setLayout(colors_layout)
        content_layout.addWidget(colors_group)
        
        # 物体检测按钮组
        objects_group = QGroupBox("物体检测")
        objects_layout = QVBoxLayout()
        objects_layout.setContentsMargins(10, 10, 10, 10)
        self.btnObjects = QPushButton("物体检测")
        self.btnObjects.setMinimumHeight(40)  # 设置最小高度
        self.btnObjects.clicked.connect(self.on_objects_button_clicked)
        
        objects_layout.addWidget(self.btnObjects)
        objects_group.setLayout(objects_layout)
        content_layout.addWidget(objects_group)
        
        # 镜头尺度分析按钮组
        shotscale_group = QGroupBox("镜头尺度分析")
        shotscale_layout = QVBoxLayout()
        shotscale_layout.setContentsMargins(10, 10, 10, 10)
        self.btnShotscale = QPushButton("镜头尺度分析")
        self.btnShotscale.setMinimumHeight(40)  # 设置最小高度
        self.btnShotscale.clicked.connect(self.on_shotscale_button_clicked)
        
        shotscale_layout.addWidget(self.btnShotscale)
        shotscale_group.setLayout(shotscale_layout)
        content_layout.addWidget(shotscale_group)
        
        # 字幕识别按钮组
        subtitle_group = QGroupBox("字幕识别")
        subtitle_layout = QVBoxLayout()
        subtitle_layout.setContentsMargins(10, 10, 10, 10)
        subtitle_label = QLabel("识别阈值")
        self.subtitleSlider = QSlider(Qt.Horizontal)
        self.subtitleSlider.setRange(40, 90)
        self.subtitleSlider.setValue(60)
        self.subtitleSlider.setTickPosition(QSlider.TicksBelow)
        self.subtitleSlider.setTickInterval(10)
        self.btnSubtitle = QPushButton("字幕识别")
        self.btnSubtitle.setMinimumHeight(40)  # 设置最小高度
        self.btnSubtitle.clicked.connect(self.on_subtitle_button_clicked)
        
        subtitle_layout.addWidget(subtitle_label)
        subtitle_layout.addWidget(self.subtitleSlider)
        subtitle_layout.addWidget(self.btnSubtitle)
        subtitle_group.setLayout(subtitle_layout)
        content_layout.addWidget(subtitle_group)
        
        # 添加底部空白，确保可以滚动到底部按钮
        spacer = QWidget()
        spacer.setMinimumHeight(20)
        content_layout.addWidget(spacer)
        
        # 将内容widget设置到滚动区域
        scroll_area.setWidget(content_widget)
        
        # 将滚动区域添加到主布局
        main_layout.addWidget(scroll_area)
        
        # 设置最小宽度，使按钮更容易点击
        content_widget.setMinimumWidth(230)
        
        # 设置控制面板的最小大小，确保滚动条能正常显示
        self.setMinimumWidth(250)
        self.setMinimumHeight(400)
        
        # 设置控制面板的主widget
        self.setWidget(main_widget)

    def on_filename_changed(self, filename):
        self.filename = filename
        
    def colorChange(self, value):
        self.colorsC = value
        self.labelColors.setText(str(value))
        
    def subtitleChange(self, value):
        self.subtitleValue = value
        self.labelSubtitlevalue.setText(str(value))
        
    def shotcut_transNetV2(self):
        """分镜头检测"""
        if not self.filename:
            QMessageBox.warning(self, "错误", "请先加载视频文件!")
            return
            
        v_path = self.filename  # 视频路径
        base_name = os.path.basename(v_path)
        if "." in base_name:
            base_name = base_name.rsplit(".", 1)[0]
            
        frame_save = f"./img/{base_name}/frame"  # 图片存储路径
        image_save = f"./img/{base_name}"
        
        # 确保目录存在
        os.makedirs(frame_save, exist_ok=True)
        
        # 使用进度对话框处理
        self.parent.start_process("shotcut", 
                                v_path=v_path,
                                image_save=image_save, 
                                th=self.th)

    def colorAnalyze(self):
        """色彩分析"""
        if self.parent.selectedVideoFile:
            # 获取视频文件名作为目录名
            video_name = os.path.basename(self.parent.selectedVideoFile).split('.')[0]
            # 构建img/视频名 目录路径
            img_path = os.path.join("./img", video_name)
            # 确保目录存在
            os.makedirs(img_path, exist_ok=True)
            os.makedirs(os.path.join(img_path, "frame"), exist_ok=True)
            
            # 启动色彩分析任务
            self.parent.start_process("color", 
                                    input_path=img_path,
                                    v_path=self.parent.selectedVideoFile,
                                    image_save=True, 
                                    frame_save=True,
                                    colors_count=self.colorsC.value())
        else:
            QMessageBox.warning(self, "警告", "请先加载视频文件")

    def object_detect(self):
        """物体检测"""
        if not self.filename:
            QMessageBox.warning(self, "错误", "请先加载视频文件!")
            return
        
        # 检查模型文件是否存在
        from algorithms.objectDetection import ObjectDetection
        obj_detect = ObjectDetection("./img")
        missing_files = obj_detect._check_model_files()
        
        if missing_files:
            QMessageBox.warning(
                self, "缺少模型文件",
                f"缺少必要的YOLO模型文件: {', '.join(missing_files)}\n"
                f"请确保这些文件已放置在 {obj_detect.models_dir} 目录中。"
            )
            return
        
        imgpath = os.path.basename(self.filename).rsplit(".", 1)[0]
        
        # 使用进度对话框处理
        self.parent.start_process("object", imgpath=imgpath)

    def getsubtitles(self):
        """执行字幕识别"""
        # 检查是否已加载视频文件
        if self.parent.selectedVideoFile:
            imgpath = os.path.basename(self.parent.selectedVideoFile).split('.')[0]

            # 启动处理
            self.parent.start_process("subtitle", 
                                    input_path=self.parent.selectedVideoFile,
                                    image_save=False, 
                                    frame_save=False,
                                    subtitle_value=self.subtitleValue)
        else:
            QMessageBox.warning(self, "警告", "请先加载视频文件")

    def show_easyocr_info(self):
        """显示EasyOCR信息"""
        guide = (
            "关于EasyOCR字幕识别：\n\n"
            "PyCinemetrics使用EasyOCR进行字幕识别。EasyOCR是一个基于深度学习的OCR系统，支持多种语言，包括中文和英文。\n\n"
            "首次使用时，系统会自动下载所需的语言模型，这可能需要一些时间。\n\n"
            "语言模型存储位置：\n"
            "- ./models/EasyOCR 目录\n\n"
            "调整字幕识别参数：\n"
            "- 使用滑块调整字幕提取的帧间隔\n"
            "- 较小的值可以提高字幕识别的准确度，但会增加处理时间\n"
            "- 较大的值处理速度更快，但可能会错过一些快速变化的字幕\n\n"
            "如果遇到识别问题，请确保视频中的字幕清晰可见。"
        )
        
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("EasyOCR 字幕识别说明")
        msg_box.setText(guide)
        msg_box.setTextFormat(Qt.TextFormat.RichText)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec()

    def getshotscale(self):
        """镜头尺度分析"""
        if self.parent.selectedVideoFile:
            # 获取视频文件名作为目录名
            video_name = os.path.basename(self.parent.selectedVideoFile).split('.')[0]
            # 构建img/视频名 目录路径
            img_path = os.path.join("./img", video_name)
            # 确保目录存在
            os.makedirs(img_path, exist_ok=True)
            os.makedirs(os.path.join(img_path, "frame"), exist_ok=True)
            
            # 启动镜头尺度分析任务
            self.parent.start_process("shotscale", 
                                    input_path=img_path,
                                    v_path=self.parent.selectedVideoFile,
                                    image_save=True, 
                                    frame_save=True)
        else:
            QMessageBox.warning(self, "警告", "请先加载视频文件")

    def show_save_dialog(self, file_name):
        """显示保存对话框"""
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        directory = QFileDialog.getExistingDirectory(
            self, "选择保存目录", "", options=options
        )

        if directory:
            self.download_resources(directory, file_name)

    def download_resources(self, directory, file_name):
        """将文件保存到指定目录"""
        try:
            if not self.filename:
                QMessageBox.warning(self, "错误", "请先加载视频文件!")
                return
                
            imgpath = os.path.basename(self.filename)[0:-4]
            source_path = f"./img/{imgpath}/{file_name}"
            
            if not os.path.exists(source_path):
                QMessageBox.warning(self, "错误", f"文件 {file_name} 不存在！请先执行相应的分析操作。")
                return
                
            import shutil
            target_path = os.path.join(directory, file_name)
            shutil.copy2(source_path, target_path)
            
            QMessageBox.information(self, "下载成功", f"文件已保存到: {target_path}")
        except Exception as e:
            QMessageBox.critical(self, "下载失败", f"下载文件时出错: {str(e)}")

    def on_shotcut_detection_button_clicked(self):
        """当用户点击分镜头检测按钮时的处理"""
        value = self.shotcutSlider.value()
        self.th = value / 1000  # 设置阈值
        self.shotcut_transNetV2()  # 调用实际的分镜头检测方法
        
    def on_colors_button_clicked(self):
        """当用户点击色彩分析按钮时的处理"""
        self.colors_count = self.colorsC.value()  # 获取当前滑块值并存储
        self.colorAnalyze()  # 调用实际的色彩分析方法
        
    def on_objects_button_clicked(self):
        """当用户点击物体检测按钮时的处理"""
        self.object_detect()  # 调用实际的物体检测方法
        
    def on_subtitle_button_clicked(self):
        """字幕识别按钮点击处理"""
        # 弹出菜单
        menu = QMenu(self)
        start_action = menu.addAction("开始字幕识别")
        info_action = menu.addAction("字幕识别说明")
        
        # 显示菜单并处理选择
        action = menu.exec_(self.btnSubtitle.mapToGlobal(QPoint(0, self.btnSubtitle.height())))
        
        if action == start_action:
            # 开始字幕识别
            self.subtitleValue = self.subtitleSlider.value()
            self.getsubtitles()  # 调用实际的字幕识别方法
        elif action == info_action:
            # 显示字幕识别信息
            self.show_easyocr_info()
        
    def on_shotscale_button_clicked(self):
        """当用户点击镜头尺度分析按钮时的处理"""
        self.getshotscale()  # 调用实际的镜头尺度分析方法