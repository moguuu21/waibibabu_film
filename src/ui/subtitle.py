import os
from PySide6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QTextEdit,
    QPushButton, QHBoxLayout, QFileDialog, QMessageBox
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QSize, Signal
import cv2
import numpy as np


class Subtitle(QDockWidget):
    def __init__(self, parent=None):
        super().__init__('字幕', parent)
        self.parent = parent
        self.filename = None
        self.init_ui()
        self.parent.filename_changed.connect(self.on_filename_changed)

    def init_ui(self):
        widget = QWidget()
        layout = QVBoxLayout()

        # 创建文本编辑区
        self.textSubtitle = QTextEdit()
        self.textSubtitle.setReadOnly(True)
        self.textSubtitle.setPlaceholderText("在这里显示识别到的字幕")
        
        # 添加按钮布局
        button_layout = QHBoxLayout()
        
        # 添加保存按钮
        self.save_button = QPushButton("保存SRT文件")
        self.save_button.clicked.connect(self.save_subtitle)
        
        # 添加导出按钮
        self.export_button = QPushButton("导出字幕文本")
        self.export_button.clicked.connect(self.export_text)
        
        # 添加清空按钮
        self.clear_button = QPushButton("清空字幕")
        self.clear_button.clicked.connect(self.clear_subtitle)
        
        # 将按钮添加到按钮布局
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.export_button)
        button_layout.addWidget(self.clear_button)
        
        # 将文本区和按钮区添加到主布局
        layout.addWidget(self.textSubtitle)
        layout.addLayout(button_layout)
        
        widget.setLayout(layout)
        self.setWidget(widget)

    def on_filename_changed(self, filename):
        self.filename = filename
        # 清空字幕文本区
        self.textSubtitle.clear()
    
    def update_subtitle(self, subtitle_text):
        """更新字幕文本"""
        self.textSubtitle.setPlainText(subtitle_text)
    
    def save_subtitle(self):
        """保存SRT字幕文件"""
        if not self.filename:
            QMessageBox.warning(self, "错误", "没有加载视频文件")
            return
            
        if self.textSubtitle.toPlainText().strip() == "":
            QMessageBox.warning(self, "错误", "没有字幕内容可保存")
            return
            
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(
            self, "保存SRT文件", "", "SRT字幕文件 (*.srt);;所有文件 (*)", options=options
        )
        
        if file_name:
            # 检查文件扩展名
            if not file_name.lower().endswith('.srt'):
                file_name += '.srt'
                
            try:
                # 获取视频所在目录的SRT文件
                video_base = os.path.basename(self.filename)
                video_name = os.path.splitext(video_base)[0]
                srt_source = f"./img/{video_name}/subtitle.srt"
                
                if os.path.exists(srt_source):
                    # 复制SRT文件
                    import shutil
                    shutil.copy2(srt_source, file_name)
                    QMessageBox.information(self, "保存成功", f"字幕文件已保存到: {file_name}")
                else:
                    # 如果找不到SRT文件，就从文本区创建一个简单的SRT文件
                    with open(file_name, 'w', encoding='utf-8') as f:
                        lines = self.textSubtitle.toPlainText().split('\n')
                        for i, line in enumerate(lines):
                            if line.strip():
                                f.write(f"{i+1}\n")
                                f.write("00:00:00,000 --> 00:00:05,000\n")
                                f.write(f"{line}\n\n")
                    QMessageBox.information(self, "保存成功", f"简单字幕文件已保存到: {file_name}")
            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"保存字幕文件时出错: {str(e)}")
    
    def export_text(self):
        """导出纯文本字幕"""
        if self.textSubtitle.toPlainText().strip() == "":
            QMessageBox.warning(self, "错误", "没有字幕内容可导出")
            return
            
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(
            self, "导出字幕文本", "", "文本文件 (*.txt);;所有文件 (*)", options=options
        )
        
        if file_name:
            # 检查文件扩展名
            if not file_name.lower().endswith('.txt'):
                file_name += '.txt'
                
            try:
                with open(file_name, 'w', encoding='utf-8') as f:
                    f.write(self.textSubtitle.toPlainText())
                QMessageBox.information(self, "导出成功", f"字幕文本已导出到: {file_name}")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"导出字幕文本时出错: {str(e)}")
    
    def clear_subtitle(self):
        """清空字幕文本"""
        if self.textSubtitle.toPlainText().strip() != "":
            reply = QMessageBox.question(self, "确认", "确定要清空当前字幕内容吗？", 
                                       QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.textSubtitle.clear()
