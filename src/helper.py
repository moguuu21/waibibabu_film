import sys
import os
import platform
import importlib
import time
from PySide6.QtWidgets import QSplashScreen, QWidget, QLabel, QVBoxLayout, QProgressBar, QApplication, QGraphicsDropShadowEffect
from PySide6.QtGui import QPixmap, QColor, QPainter, QPen, QFont, QLinearGradient, QGuiApplication
from PySide6.QtCore import Qt, QTimer, QCoreApplication, QPoint, QSize, QRect

def resource_path(relative_path):
    myos = platform.system()

    if (myos == 'Windows'):
        try:
            base_path = sys._MEIPASS
        except AttributeError:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)
    elif (myos == 'Darwin') or (myos == 'Linux'):
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(
            os.path.abspath(__file__)))
        return os.path.join(base_path, relative_path)
    return None


class Splash(QSplashScreen):
    
    def __init__(self, splash_img=''):
        super().__init__()
        self.setModal = True
        self._init_splash_screen()
        self.loading_steps=[]
        
        # 创建定时器用于动画效果
        self.timer = QTimer(self)
        self.timer.timeout.connect(lambda: self.update_progress(self.timer.interval() % 100))
        self.timer.start(50)  # 每50毫秒更新一次
    
    def _init_splash_screen(self):
        # 使用一个QLabel创建自定义格式的splash screen
        self.surface = QLabel()
        # 设置无边框窗口
        self.surface.setWindowFlags(Qt.FramelessWindowHint)
        
        # 创建渐变背景
        gradient = QLinearGradient(0, 0, 0, 350)
        gradient.setColorAt(0, QColor("#E3F2FD"))  # 浅蓝色
        gradient.setColorAt(1, QColor("#FAFBFF"))  # 接近白色的浅蓝
        
        # 绘制背景
        pixmap = QPixmap(480, 350)
        painter = QPainter(pixmap)
        painter.fillRect(0, 0, 480, 350, gradient)
        
        # 绘制顶部装饰条 - 使用品牌蓝色
        painter.fillRect(0, 0, 480, 4, QColor("#2962FF"))
        
        # 标题文字 - 使用深蓝色
        painter.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        painter.setPen(QColor("#0039CB"))
        painter.drawText(QRect(0, 30, 480, 40), Qt.AlignCenter, "PyCinemetrics")
        
        # 副标题 - 使用靛蓝色
        painter.setFont(QFont("Microsoft YaHei", 11))
        painter.setPen(QColor("#5C6BC0"))
        painter.drawText(QRect(0, 70, 480, 30), Qt.AlignCenter, "电影分析与人脸识别工具")
        
        # 绘制功能点列表
        features = [
            "分镜头检测与分析",
            "色彩分析和调色板提取",
            "智能物体检测与统计",
            "字幕识别与SRT生成",
            "镜头尺度与人物分析"
        ]
        
        feature_colors = [
            "#2962FF",  # 明亮深蓝色
            "#1E88E5",  # 蓝色
            "#0091EA",  # 浅蓝色
            "#00B0FF",  # 亮蓝色
            "#40C4FF"   # 亮青色
        ]
        
        y_pos = 120
        for i, feature in enumerate(features):
            # 绘制彩色点
            painter.setBrush(QColor(feature_colors[i]))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPoint(35, y_pos + 8), 4, 4)
            
            # 绘制功能文本
            painter.setPen(QColor("#212121"))
            painter.setFont(QFont("Microsoft YaHei", 10))
            painter.drawText(QRect(50, y_pos, 400, 20), Qt.AlignLeft | Qt.AlignVCenter, feature)
            
            y_pos += 25
        
        # 加载中文本
        painter.setFont(QFont("Microsoft YaHei", 10))
        painter.setPen(QColor("#616161"))
        painter.drawText(QRect(0, 245, 480, 20), Qt.AlignCenter, "加载中...")
        
        # 进度条背景
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#E0E0E0"))
        painter.drawRoundedRect(90, 280, 300, 10, 5, 5)
        
        # 进度条（初始为0%）
        painter.setBrush(QColor("#2979FF"))
        painter.drawRoundedRect(90, 280, 0, 10, 5, 5)
        
        # 版本信息
        painter.setFont(QFont("Microsoft YaHei", 8))
        painter.setPen(QColor("#9E9E9E"))
        painter.drawText(QRect(0, 320, 480, 20), Qt.AlignCenter, "版本 1.0.0")
        
        painter.end()
        
        # 设置阴影效果
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 2)
        self.surface.setGraphicsEffect(shadow)
        
        # 设置pixmap到surface
        self.surface.setPixmap(pixmap)
        
        # 保存pixmap用于更新进度
        self.pixmap = pixmap
        
        # 将surface加入splash screen
        self.setPixmap(pixmap)
        self.setMask(pixmap.mask())
        
        # 居中显示
        rect = QGuiApplication.primaryScreen().geometry()
        self.move(int((rect.width() - self.width()) / 2), int((rect.height() - self.height()) / 2))
        
        # 保存进度条位置和大小
        self.progress_x = 90
        self.progress_y = 280
        self.progress_width = 300
        self.progress_height = 10
    
    def update_text(self, text):
        """更新加载文本"""
        # 创建临时pixmap并重绘
        pixmap = self.pixmap.copy()
        painter = QPainter(pixmap)
        
        # 擦除原文本区域
        gradient = QLinearGradient(0, 245, 0, 265)
        gradient.setColorAt(0, QColor("#E3F2FD"))
        gradient.setColorAt(1, QColor("#FAFBFF"))
        painter.fillRect(0, 245, 480, 20, gradient)
        
        # 绘制新文本
        painter.setFont(QFont("Microsoft YaHei", 10))
        painter.setPen(QColor("#616161"))
        painter.drawText(QRect(0, 245, 480, 20), Qt.AlignCenter, text)
        
        painter.end()
        
        # 更新splash screen
        self.setPixmap(pixmap)
        
        # 保存更新后的pixmap
        self.pixmap = pixmap
    
    def update_progress(self, progress, text=None):
        """更新进度条"""
        if text:
            self.update_text(text)
            
        if progress < 0:
            progress = 0
        if progress > 100:
            progress = 100
        
        # 计算进度条宽度
        width = int(self.progress_width * progress / 100)
        
        # 创建临时pixmap并重绘
        pixmap = self.pixmap.copy()
        painter = QPainter(pixmap)
        
        # 擦除原进度条
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#E0E0E0"))
        painter.drawRoundedRect(self.progress_x, self.progress_y, self.progress_width, self.progress_height, 5, 5)
        
        # 绘制新进度条
        # 使用渐变色的进度条
        gradient = QLinearGradient(self.progress_x, 0, self.progress_x + self.progress_width, 0)
        gradient.setColorAt(0, QColor("#1976D2"))  # 蓝色
        gradient.setColorAt(1, QColor("#2979FF"))  # 亮蓝色
        
        painter.setBrush(gradient)
        painter.drawRoundedRect(self.progress_x, self.progress_y, width, self.progress_height, 5, 5)
        
        painter.end()
        
        # 更新splash screen
        self.setPixmap(pixmap)
        
        # 保存更新后的pixmap
        self.pixmap = pixmap

    def finish(self, widget):
        """完成启动过程"""
        # 停止计时器
        self.timer.stop()
        
        # 确保没有正在进行的绘图操作
        QCoreApplication.processEvents()
        
        # 等待一小段时间确保绘图操作完成
        time.sleep(0.1)
        
        try:
            # 创建最终的完成状态图像
            final_pixmap = self.pixmap.copy()
            
            # 在副本上绘制完成的进度条
            painter = QPainter(final_pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # 创建渐变进度条
            progress_gradient = QLinearGradient(self.progress_x, 0, self.progress_x + self.progress_width, 0)
            progress_gradient.setColorAt(0, QColor("#2962FF"))  # 明亮深蓝色
            progress_gradient.setColorAt(1, QColor("#00B8D4"))  # 绿松石色
            
            painter.setPen(Qt.NoPen)
            painter.setBrush(progress_gradient)
            painter.drawRoundedRect(self.progress_x, self.progress_y, self.progress_width, self.progress_height, 5, 5)
            
            # 更新加载完成的文本
            painter.setPen(QPen(QColor("#0039CB")))
            loading_font = QFont("Microsoft YaHei", 10, QFont.Bold)
            painter.setFont(loading_font)
            painter.drawText(self.progress_x + 10, self.progress_y + 5, "加载完成，准备就绪!")
            
            # 添加完成勾号图标
            check_x = self.progress_x + self.progress_width - 10
            check_y = self.progress_y + 5
            
            painter.setPen(QPen(QColor("#00B8D4"), 3))
            painter.drawLine(check_x - 5, check_y - 2, check_x - 2, check_y + 2)
            painter.drawLine(check_x - 2, check_y + 2, check_x + 5, check_y - 5)
            
            painter.end()
            
            # 更新显示
            self.setPixmap(final_pixmap)
            
            # 短暂显示完成状态
            self.repaint()
            time.sleep(0.8)
        except Exception as e:
            print(f"注意: 在完成启动画面时发生错误: {e}")
        
        # 完成启动
        super().finish(widget)
