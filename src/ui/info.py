import os
import functools
from PySide6.QtWidgets import (QLabel, QPushButton, QHBoxLayout, QApplication, 
                               QWidget, QFileDialog, QVBoxLayout, QMainWindow, 
                               QGridLayout, QTableWidget, QTableWidgetItem, 
                               QLineEdit, QDockWidget, QProgressBar)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Signal, Qt, QTimer
import csv
import re

class Info(QDockWidget):
    video_info_loaded = Signal()

    def __init__(self, parent=None):
        super().__init__('视频信息', parent)
        self.parent = parent
        self.init_ui()
        self.parent.shot_finished.connect(self.update_table)
        
    def init_ui(self):
        widget = QWidget()
        main_layout = QVBoxLayout()
        
        # 创建表格
        self.table_widget = QTableWidget(0, 2)
        self.table_widget.setHorizontalHeaderLabels(['属性', '值'])
        self.table_widget.horizontalHeader().setStretchLastSection(True)
        self.table_widget.setStyleSheet("QTableWidget {border: none;}")
        
        # 视频信息字典
        self.properties = [
            ("文件名", ""),
            ("分辨率", ""),
            ("时长", ""),
            ("帧率", ""),
            ("帧数", ""),
            ("比特率", ""),
            ("编解码器", ""),
            ("尺寸", ""),
            ("字幕", "无"),
            ("语言", "未知"),
            ("声道", ""),
        ]
        
        # 初始化表格
        self.update_table()
        
        main_layout.addWidget(self.table_widget)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        
        self.load_video_button = QPushButton("打开视频", self)
        self.load_video_button.clicked.connect(self.load_video)
        button_layout.addWidget(self.load_video_button)
        
        main_layout.addLayout(button_layout)
        
        widget.setLayout(main_layout)
        self.setWidget(widget)
        
    def update_table(self):
        # 更新表格内容
        self.table_widget.setRowCount(0)
        
        for row, (prop, value) in enumerate(self.properties):
            self.table_widget.insertRow(row)
            self.table_widget.setItem(row, 0, QTableWidgetItem(prop))
            self.table_widget.setItem(row, 1, QTableWidgetItem(value))
            
        self.table_widget.resizeColumnsToContents()
    
    def load_video(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv);;所有文件 (*)", options=options
        )
        
        if filename:
            self.parent.filename_changed.emit(filename)
            # 更新视频信息
            self.update_video_info(filename)
    
    def update_video_info(self, filename):
        """更新视频信息"""
        try:
            import vlc
            import time
            
            # 创建VLC实例和媒体
            instance = vlc.Instance()
            media = instance.media_new(filename)
            media.parse()
            
            # 获取基本信息
            base_name = os.path.basename(filename)
            duration_ms = media.get_duration()
            duration_str = time.strftime('%H:%M:%S', time.gmtime(duration_ms/1000))
            
            # 查找视频轨道信息
            width = height = fps = frame_count = bitrate = codec = channels = 0
            
            # 获取视频轨道信息
            tracks_info = media.tracks_get()
            for track in tracks_info:
                if track.type == vlc.TrackType.video:
                    width = track.video.width
                    height = track.video.height
                    fps = round(track.video.frame_rate, 2)
                    frame_count = int(fps * (duration_ms / 1000))
                    codec = track.codec_name
                    
                if track.type == vlc.TrackType.audio:
                    channels = track.audio.channels
                    
            # 检查比特率
            if hasattr(media, 'get_bitrate'):
                bitrate = media.get_bitrate() // 1000  # 转换为kbps
            else:
                # 估计比特率
                file_size = os.path.getsize(filename)
                bitrate = int(file_size * 8 / (duration_ms / 1000) / 1000)  # kbps
                
            # 更新属性
            new_properties = [
                ("文件名", base_name),
                ("分辨率", f"{width}x{height}"),
                ("时长", duration_str),
                ("帧率", f"{fps} fps"),
                ("帧数", str(frame_count)),
                ("比特率", f"{bitrate} kbps"),
                ("编解码器", codec),
                ("尺寸", f"{self.format_file_size(os.path.getsize(filename))}"),
                ("字幕", "无" if media.has_subtitles() == 0 else "有"),
                ("语言", "未知"),
                ("声道", f"{channels}"),
            ]
            
            self.properties = new_properties
            self.update_table()
            self.video_info_loaded.emit()
            
        except Exception as e:
            print(f"更新视频信息时出错: {e}")
            # 基本的文件信息
            base_name = os.path.basename(filename)
            file_size = os.path.getsize(filename)
            
            self.properties = [
                ("文件名", base_name),
                ("分辨率", "未知"),
                ("时长", "未知"),
                ("帧率", "未知"),
                ("帧数", "未知"),
                ("比特率", "未知"),
                ("编解码器", "未知"),
                ("尺寸", f"{self.format_file_size(file_size)}"),
                ("字幕", "未知"),
                ("语言", "未知"),
                ("声道", "未知"),
            ]
            self.update_table()
    
    def format_file_size(self, size_bytes):
        """格式化文件大小为人类可读形式"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes/1024:.2f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes/(1024*1024):.2f} MB"
        else:
            return f"{size_bytes/(1024*1024*1024):.2f} GB"
