import os
from PySide6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox
)
from PySide6.QtGui import QPixmap, QImage, QIcon
from PySide6.QtCore import Qt, Signal, QSize
import cv2


class Analyze(QDockWidget):
    def __init__(self, parent=None):
        super().__init__('分析结果', parent)
        self.parent = parent
        self.init_ui()
        
    def init_ui(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 创建结果图像显示区域
        self.labelAnalyze = QLabel()
        self.labelAnalyze.setAlignment(Qt.AlignCenter)
        self.labelAnalyze.setMinimumSize(260, 180)
        self.labelAnalyze.setStyleSheet("""
            QLabel {
                background-color: #F5F5F5;
                border: 1px solid #E0E0E0;
                border-radius: 4px;
            }
        """)
        self.labelAnalyze.setText("分析结果将显示在这里")
        
        # 创建按钮区域
        button_layout = QHBoxLayout()
        
        # 保存按钮
        self.save_button = QPushButton("保存图像")
        self.save_button.clicked.connect(self.save_image)
        
        # 查看大图按钮
        self.view_button = QPushButton("查看大图")
        self.view_button.clicked.connect(self.view_full_image)
        
        # 添加按钮到布局
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.view_button)
        
        # 将控件添加到主布局
        layout.addWidget(self.labelAnalyze)
        layout.addLayout(button_layout)
        
        widget.setLayout(layout)
        self.setWidget(widget)
        
    def save_image(self):
        """保存分析结果图像"""
        if not hasattr(self.parent, 'AnalyzeImgPath') or not self.parent.AnalyzeImgPath:
            QMessageBox.warning(self, "错误", "没有可保存的分析图像")
            return
            
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(
            self, "保存图像", "", "PNG图像 (*.png);;JPEG图像 (*.jpg);;所有文件 (*)", options=options
        )
        
        if file_name:
            # 确保有正确的扩展名
            if not (file_name.lower().endswith('.png') or file_name.lower().endswith('.jpg')):
                file_name += '.png'
                
            try:
                pixmap = QPixmap(self.parent.AnalyzeImgPath)
                if not pixmap.isNull():
                    pixmap.save(file_name)
                    QMessageBox.information(self, "保存成功", f"图像已保存到: {file_name}")
                else:
                    QMessageBox.warning(self, "保存失败", "无法加载图像")
            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"保存图像时出错: {str(e)}")

    def view_full_image(self):
        """查看完整大图"""
        if not hasattr(self.parent, 'AnalyzeImgPath') or not self.parent.AnalyzeImgPath:
            QMessageBox.warning(self, "错误", "没有可查看的分析图像")
            return

        try:
            from PySide6.QtWidgets import QDialog, QScrollArea, QPushButton
            from PySide6.QtGui import QPixmap,QGuiApplication
            from PySide6.QtCore import Qt, QSize

            dialog = QDialog(self)
            dialog.setWindowTitle("分析结果 - 完整视图")

            # 加载图片
            pixmap = QPixmap(self.parent.AnalyzeImgPath)
            if pixmap.isNull():
                QMessageBox.warning(self, "查看失败", "无法加载图像")
                return

            # 获取屏幕可用尺寸
            screen = QGuiApplication.primaryScreen()
            screen_geometry = screen.availableGeometry()
            screen_width = screen_geometry.width()
            screen_height = screen_geometry.height()

            # 获取图片原始尺寸
            original_width = pixmap.width()
            original_height = pixmap.height()

            # 设置窗口边距和按钮高度
            margin = 20  # 窗口边距
            button_height = 40  # 按钮高度
            titlebar_height = 30  # 标题栏高度

            # 计算可用的最大内容区域尺寸
            max_content_width = screen_width - margin * 2
            max_content_height = screen_height - margin * 2 - button_height - titlebar_height

            # 计算缩放比例
            width_ratio = max_content_width / original_width
            height_ratio = max_content_height / original_height
            scale_factor = min(width_ratio, height_ratio, 1.2)  # 最大放大1.2倍

            # 应用缩放
            if scale_factor < 1.0:  # 图片大于可用区域，缩小
                scaled_width = int(original_width * scale_factor)
                scaled_height = int(original_height * scale_factor)
                pixmap = pixmap.scaled(scaled_width, scaled_height,
                                       Qt.KeepAspectRatio, Qt.SmoothTransformation)
            elif scale_factor > 1.0:  # 图片小于可用区域，放大但不超过1.2倍
                scaled_width = int(original_width * scale_factor)
                scaled_height = int(original_height * scale_factor)
                pixmap = pixmap.scaled(scaled_width, scaled_height,
                                       Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # 计算最终窗口尺寸
            final_width = pixmap.width() + margin * 2
            final_height = pixmap.height() + margin * 2 + button_height

            # 设置窗口大小
            dialog.setMinimumSize(final_width, final_height)
            dialog.setMaximumSize(final_width, final_height)

            # 创建布局
            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(margin, margin, margin, margin)
            layout.setSpacing(10)

            # 创建QLabel并显示图片
            image_label = QLabel()
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setPixmap(pixmap)
            layout.addWidget(image_label)

            # 添加关闭按钮
            close_button = QPushButton("关闭")
            close_button.setFixedHeight(button_height)
            close_button.clicked.connect(dialog.accept)
            layout.addWidget(close_button)

            # 居中显示
            dialog.move((screen_width - final_width) // 2, (screen_height - final_height) // 2)

            dialog.exec()

        except Exception as e:
            QMessageBox.critical(self, "查看失败", f"显示图像时出错: {str(e)}")