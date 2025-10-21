import os
import sys
# from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QMessageBox, QProgressBar,
    QStyleFactory, QTabWidget, QProgressDialog
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QPalette, QColor, QAction, QPixmap
from ui.timeline import Timeline
from ui.info import Info
from ui.analyze import Analyze
from ui.subtitle import Subtitle
from ui.face_recognition_ui import FaceRecognitionUI
from qt_material import apply_stylesheet
from concurrent.futures import ThreadPoolExecutor
from helper import Splash
from ui.vlcplayer import VLCPlayer
from ui.control import Control
from ProcessThread import ProcessThread  # 导入单独的线程处理类
# os.chdir(os.path.dirname(os.path.abspath(__file__)))


# from ui.subtitleEasyOcr import getsubtitleEasyOcr,subtitle2Srt

# 删除重复的ProcessThread类定义，使用导入的类

class ProgressDialog(QProgressDialog):
    """进度对话框类，用于显示和更新任务进度"""
    
    def __init__(self, parent=None):
        """初始化进度对话框
        
        Args:
            parent: 父窗口
        """
        super(ProgressDialog, self).__init__(parent)
        self.setWindowTitle("处理中...")
        self.setLabelText("正在准备处理...")
        self.setRange(0, 100)
        self.setValue(0)
        self.setMinimumDuration(0)
        self.setWindowModality(Qt.WindowModal)
        self.setMinimumWidth(300)
        
        # 设置不可关闭
        self.setCancelButton(None)
        
    def update_progress(self, value, message=None):
        """更新进度值和消息
        
        Args:
            value: 进度值（0-100）
            message: 进度消息
        """
        self.setValue(value)
        if message:
            self.setLabelText(message)

class MainWindow(QMainWindow):
    filename_changed = Signal(str)
    shot_finished = Signal()
    video_play_changed= Signal(int)

    def __init__(self):
        super().__init__()
        self.threadpool = ThreadPoolExecutor()
        self.filename = ''
        self.process_thread = None
        self.progress_dialog = None

        self.AnalyzeImgPath=''
        self.init_ui()
        
        # 应用自定义样式
        self.apply_custom_style()

    def init_ui(self):
        #self.setWindowIcon(QIcon(resource_path('resources/icon.ico')))

        # Delay VLC import to here
        self.player = VLCPlayer(self)
        self.setCentralWidget(self.player)

        self.info = Info(self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.info)

        self.subtitle = Subtitle(self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.subtitle)

        self.control = Control(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.control)

        # 添加timeline
        self.colorc = self.control.colorsC
        self.timeline = Timeline(self, self.colorc)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.timeline)

        # 添加人脸识别UI
        self.face_recognition_ui = FaceRecognitionUI(self)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.face_recognition_ui)

        # 添加analyze dockWidget
        self.analyze = Analyze(self)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.analyze)
        
        # 设置底部Dock窗口的位置关系，确保人脸识别面板位于timeline和analyze之间
        self.tabifyDockWidget(self.timeline, self.face_recognition_ui)
        self.tabifyDockWidget(self.face_recognition_ui, self.analyze)
        
        # 确保timeline是默认可见的标签页
        self.timeline.raise_()

        # 设置窗口属性
        self.setWindowTitle("PyCinemetrics - 电影分析工具")
        self.setGeometry(100, 100, 1280, 800)
        
        # 禁止区分同类型dockwidget tab化
        self.setDockNestingEnabled(True)
        self.setTabPosition(Qt.AllDockWidgetAreas, QTabWidget.North)
        
        # 连接信号
        self.filename_changed.connect(self.on_filename_changed)
        self.filename_changed.connect(self.subtitle.on_filename_changed)
        self.filename_changed.connect(self.control.on_filename_changed)
        self.video_play_changed.connect(self.player.play_specific_frame)

        # 添加拖放支持
        self.setAcceptDrops(True)
        
    def apply_custom_style(self):
        """应用自定义界面样式 - 蓝白配色升级版"""
        # 使用qt-material库设置主题
        apply_stylesheet(self, theme='light_blue.xml', invert_secondary=True,
                       extra={
                           # 主色调 - 明亮的蓝色
                           'primary': '#1976D2',
                           'primary_light': '#42A5F5',
                           'primary_dark': '#0D47A1',
                           # 次要色调 - 白色/浅灰
                           'secondary': '#FFFFFF',
                           'secondary_light': '#F5F5F5',
                           'secondary_dark': '#E0E0E0',
                           # 强调色
                           'accent': '#2979FF',
                           # 背景色
                           'background': '#FAFBFF',
                           # 文本颜色
                           'text': '#212121',
                           'text_light': '#757575',
                           # 错误颜色
                           'error': '#D32F2F',
                       })
        
        # 更复杂的样式表，添加渐变、阴影等现代UI元素
        self.setStyleSheet(self.styleSheet() + """
            /* 窗口样式 */
            QMainWindow {
                background-color: #FAFBFF;
            }
            
            /* 标签样式 */
            QLabel {
                color: #212121;
                font-size: 12px;
            }
            
            /* 按钮样式 */
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 #42A5F5, stop:1 #1976D2);
                color: white;
                border-radius: 5px;
                padding: 5px 10px;
                font-weight: bold;
                border: none;
                min-height: 25px;
            }
            
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 #64B5F6, stop:1 #42A5F5);
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            }
            
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 #0D47A1, stop:1 #1565C0);
            }
            
            /* 进度条样式 */
            QProgressBar {
                border: 1px solid #E0E0E0;
                border-radius: 4px;
                background-color: #F5F5F5;
                text-align: center;
                color: #212121;
                height: 12px;
            }
            
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                            stop:0 #42A5F5, stop:1 #2979FF);
                border-radius: 3px;
            }
            
            /* 列表样式 */
            QListWidget {
                background-color: white;
                border: 1px solid #E0E0E0;
                border-radius: 5px;
                alternate-background-color: #F5F5F5;
            }
            
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #F0F0F0;
            }
            
            QListWidget::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 #E3F2FD, stop:1 #BBDEFB);
                color: #0D47A1;
                border-left: 3px solid #2979FF;
            }
            
            /* 滑块样式 */
            QSlider::groove:horizontal {
                height: 8px;
                background: #F5F5F5;
                border-radius: 4px;
                border: 1px solid #E0E0E0;
            }
            
            QSlider::handle:horizontal {
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,
                             stop:0 #FFFFFF, stop:1 #1976D2);
                border: 1px solid #1976D2;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 #BBDEFB, stop:1 #64B5F6);
                border-radius: 3px;
                border: 1px solid #1976D2;
            }
            
            /* 停靠窗口样式 */
            QDockWidget {
                font-weight: bold;
                color: #0D47A1;
                titlebar-close-icon: url(:/close.png);
                titlebar-normal-icon: url(:/float.png);
            }
            
            QDockWidget::title {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 #E3F2FD, stop:1 #BBDEFB);
                padding-left: 5px;
                padding-top: 2px;
                border-radius: 2px;
            }
            
            /* 分组框样式 */
            QGroupBox {
                border: 1px solid #BBDEFB;
                border-radius: 5px;
                margin-top: 20px;
                background-color: rgba(187, 222, 251, 0.1);
                font-weight: bold;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                color: #0D47A1;
                background-color: white;
            }
            
            /* 选项卡样式 */
            QTabBar::tab {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 #F5F5F5, stop:1 #E0E0E0);
                border: 1px solid #D0D0D0;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                min-width: 8ex;
                padding: 5px 10px;
                color: #616161;
            }
            
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 #E3F2FD, stop:1 #BBDEFB);
                border-bottom: none;
                border-top: 2px solid #1976D2;
                color: #0D47A1;
            }
            
            QTabBar::tab:hover:!selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 #E0E0E0, stop:1 #D0D0D0);
                color: #424242;
            }
            
            /* 下拉菜单样式 */
            QComboBox {
                border: 1px solid #E0E0E0;
                border-radius: 4px;
                padding: 3px 10px 3px 10px;
                background: white;
                selection-background-color: #BBDEFB;
                selection-color: #0D47A1;
            }
            
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #E0E0E0;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }
            
            QComboBox::drop-down:hover {
                background: #F5F5F5;
            }
            
            /* 工具提示样式 */
            QToolTip {
                border: 1px solid #BBDEFB;
                background-color: #E3F2FD;
                color: #0D47A1;
                padding: 5px;
                border-radius: 3px;
                opacity: 200;
            }
            
            /* 菜单样式 */
            QMenu {
                background-color: white;
                border: 1px solid #E0E0E0;
                border-radius: 3px;
            }
            
            QMenu::item {
                padding: 5px 30px 5px 30px;
                border-bottom: 1px solid #F0F0F0;
            }
            
            QMenu::item:selected {
                background-color: #BBDEFB;
                color: #0D47A1;
            }
            
            /* 滚动条样式 */
            QScrollBar:horizontal {
                border: none;
                background: #F5F5F5;
                height: 10px;
                margin: 0px 20px 0 20px;
                border-radius: 5px;
            }
            
            QScrollBar::handle:horizontal {
                background: #BBDEFB;
                min-width: 20px;
                border-radius: 5px;
            }
            
            QScrollBar::handle:horizontal:hover {
                background: #64B5F6;
            }
            
            QScrollBar::add-line:horizontal {
                border: none;
                background: #F5F5F5;
                width: 20px;
                border-top-right-radius: 5px;
                border-bottom-right-radius: 5px;
                subcontrol-position: right;
                subcontrol-origin: margin;
            }
            
            QScrollBar::sub-line:horizontal {
                border: none;
                background: #F5F5F5;
                width: 20px;
                border-top-left-radius: 5px;
                border-bottom-left-radius: 5px;
                subcontrol-position: left;
                subcontrol-origin: margin;
            }
            
            QScrollBar:vertical {
                border: none;
                background: #F5F5F5;
                width: 10px;
                margin: 20px 0 20px 0;
                border-radius: 5px;
            }
            
            QScrollBar::handle:vertical {
                background: #BBDEFB;
                min-height: 20px;
                border-radius: 5px;
            }
            
            QScrollBar::handle:vertical:hover {
                background: #64B5F6;
            }
            
            QScrollBar::add-line:vertical {
                border: none;
                background: #F5F5F5;
                height: 20px;
                border-bottom-left-radius: 5px;
                border-bottom-right-radius: 5px;
                subcontrol-position: bottom;
                subcontrol-origin: margin;
            }
            
            QScrollBar::sub-line:vertical {
                border: none;
                background: #F5F5F5;
                height: 20px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                subcontrol-position: top;
                subcontrol-origin: margin;
            }
        """)
        
    def _show_progress_dialog(self, title):
        """显示进度对话框"""
        self.progress_dialog = QProgressDialog(self)
        self.progress_dialog.setWindowTitle(title)
        self.progress_dialog.setLabelText("初始化中...")
        self.progress_dialog.setRange(0, 100)
        self.progress_dialog.setCancelButton(None)  # 不允许取消
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.setWindowFlags(self.progress_dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        
        # 自定义进度对话框样式
        self.progress_dialog.setStyleSheet("""
            QProgressDialog {
                background-color: #FAFBFF;
                border: 1px solid #BBDEFB;
                border-radius: 8px;
            }
            QLabel {
                color: #0D47A1;
                font-size: 12px;
                font-weight: bold;
                padding: 10px;
            }
            QProgressBar {
                border: 1px solid #BBDEFB;
                border-radius: 5px;
                background-color: #F5F5F5;
                text-align: center;
                color: #0D47A1;
                height: 20px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                            stop:0 #42A5F5, stop:1 #2979FF);
                border-radius: 4px;
            }
        """)
        
        self.progress_dialog.show()
        QApplication.processEvents()

    def on_filename_changed(self, filename=None):
        # self.labelAnalyze.setPixmap(QPixmap())
        if filename:
            path = os.path.abspath(filename)
            # 创建必要的目录
            img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "img")
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
                
            # 修复：直接将path值传给player.media，而不是调用open_file
            self.player.media = self.player.instance.media_new(path)
            self.player.mediaplayer.set_media(self.player.media)
            
            # 设置视频窗口
            import platform
            if platform.system() == 'Linux':
                self.player.mediaplayer.set_xwindow(int(self.player.videoframe.winId()))
            elif platform.system() == 'Windows':
                self.player.mediaplayer.set_hwnd(int(self.player.videoframe.winId()))
            elif platform.system() == 'Darwin':
                self.player.mediaplayer.set_nsobject(int(self.player.videoframe.winId()))
                
            # 播放视频
            self.player.play_pause()
            self.filename = path
            # 设置selectedVideoFile属性
            self.selectedVideoFile = path
            
    def show_progress_dialog(self, title, description):
        """显示进度对话框
        
        Args:
            title: 对话框标题
            description: 初始描述
            
        Returns:
            QProgressDialog: 创建的进度对话框
        """
        from PySide6.QtWidgets import QProgressDialog
        from PySide6.QtCore import Qt
        
        # 关闭之前的进度对话框，如果存在
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.close()
            
        # 创建新的进度对话框
        self.progress_dialog = QProgressDialog(description, "取消", 0, 100, self)
        self.progress_dialog.setWindowTitle(title)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)  # 立即显示
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.setAutoReset(True)
        self.progress_dialog.setValue(0)
        self.progress_dialog.show()
        
        # 连接取消信号
        self.progress_dialog.canceled.connect(self.cancel_process)
        
        return self.progress_dialog
        
    def cancel_process(self):
        """取消正在执行的处理"""
        if hasattr(self, 'process_thread') and self.process_thread:
            self.process_thread.stop()
            self.process_thread = None
    
    def update_progress(self, progress, message):
        """更新进度对话框
        
        Args:
            progress: 进度百分比（0-100）
            message: 进度消息
        """
        if self.progress_dialog:
            self.progress_dialog.update_progress(progress, message)
    
    def handle_process_error(self, error_msg):
        """处理过程中的错误"""
        QMessageBox.critical(self, "处理错误", error_msg)
        if self.progress_dialog:
            self.progress_dialog.reset()

    def start_process(self, task_type, **kwargs):
        """启动处理任务
        
        Args:
            task_type: 任务类型 (shotcut, color, object, subtitle, shotscale)
            **kwargs: 额外参数
        """
        # 如果已经有进度对话框正在显示，先关闭它
        if self.progress_dialog and self.progress_dialog.isVisible():
            self.progress_dialog.close()
            self.progress_dialog = None
            
        # 如果有正在运行的进程，先停止它
        if self.process_thread and self.process_thread.isRunning():
            self.process_thread.stop()
            self.process_thread.wait()
            
        # 显示进度对话框
        self.progress_dialog = ProgressDialog(self)
        self.progress_dialog.setWindowTitle("处理中")
        
        # 根据任务类型设置初始消息
        task_messages = {
            "shotcut": "正在进行镜头切分...",
            "color": "正在分析色彩...",
            "object": "正在进行物体识别...",
            "subtitle": "正在进行字幕识别...",
            "shotscale": "正在分析镜头尺度..."
        }
        
        initial_message = task_messages.get(task_type, "正在处理...")
        self.progress_dialog.setLabelText(initial_message)
        
        # 显示进度对话框
        self.progress_dialog.show()
        
        # 创建工作线程
        if 'input_path' not in kwargs and hasattr(self, 'selectedVideoFile'):
            kwargs['input_path'] = self.selectedVideoFile
        
        self.process_thread = ProcessThread(task_type=task_type, **kwargs)
        
        # 连接信号
        self.process_thread.progress_signal.connect(self.update_progress)
        self.process_thread.finish_signal.connect(self.on_process_finished)
        
        # 启动线程
        self.process_thread.start()

    def on_process_finished(self, success, message):
        """处理完成的回调"""
        # 关闭进度对话框
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        # 显示处理结果
        if success:
            QMessageBox.information(self, "处理完成", message)
            
            # 获取当前视频的基础文件名
            if self.filename:
                filename_base = os.path.basename(self.filename).split('.')[0]
                img_dir = f"img/{filename_base}"
                
                # 根据不同任务类型更新分析结果显示
                if "镜头切分完成" in message:
                    # 发送shot_finished信号
                    self.shot_finished.emit()
                    # 暂不显示镜头切分图像，因为没有单独的图片
                
                elif "色彩分析完成" in message:
                    # 显示色彩分析结果
                    color_result_path = os.path.join(img_dir, "color.png")
                    if os.path.exists(color_result_path):
                        self.AnalyzeImgPath = color_result_path
                        pixmap = QPixmap(color_result_path)
                        self.analyze.labelAnalyze.setPixmap(pixmap.scaled(250, 160, Qt.KeepAspectRatio))
                
                elif "物体检测完成" in message:
                    # 显示物体检测结果
                    object_result_path = os.path.join(img_dir, "objects.png")
                    if os.path.exists(object_result_path):
                        self.AnalyzeImgPath = object_result_path
                        pixmap = QPixmap(object_result_path)
                        self.analyze.labelAnalyze.setPixmap(pixmap.scaled(250, 160, Qt.KeepAspectRatio))
                
                elif "字幕识别完成" in message:
                    # 显示字幕识别结果
                    subtitle_result_path = os.path.join(img_dir, "subtitles_timeline.png")
                    if os.path.exists(subtitle_result_path):
                        self.AnalyzeImgPath = subtitle_result_path
                        pixmap = QPixmap(subtitle_result_path)
                        self.analyze.labelAnalyze.setPixmap(pixmap.scaled(250, 160, Qt.KeepAspectRatio))
                    
                    # 读取并显示字幕文本
                    srt_path = os.path.join(img_dir, "subtitle.srt")
                    if os.path.exists(srt_path):
                        try:
                            with open(srt_path, 'r', encoding='utf-8') as f:
                                subtitle_text = f.read()
                            # 更新到字幕面板
                            self.subtitle.update_subtitle(subtitle_text)
                        except Exception as e:
                            print(f"读取字幕文件失败: {str(e)}")
                
                elif "镜头尺度分析完成" in message:
                    # 显示镜头尺度分析结果
                    shotscale_result_path = os.path.join(img_dir, "shotscale.png")
                    if os.path.exists(shotscale_result_path):
                        self.AnalyzeImgPath = shotscale_result_path
                        pixmap = QPixmap(shotscale_result_path)
                        self.analyze.labelAnalyze.setPixmap(pixmap.scaled(250, 160, Qt.KeepAspectRatio))
        else:
            QMessageBox.warning(self, "处理失败", message)

    def on_shotcut_clicked(self, th):
        """当用户点击分镜头按钮时的处理"""
        if not self.filename:
            QMessageBox.warning(self, "警告", "请先选择视频文件")
            return
            
        # 计算保存路径
        filename_base = os.path.basename(self.filename).split('.')[0]
        image_save = f"img/{filename_base}/frame"
        
        # 启动处理线程
        self.start_process(
            "shotcut", 
            v_path=self.filename, 
            image_save=image_save, 
            th=th
        )
    
    def on_colors_clicked(self, colors_count):
        """当用户点击色彩分析按钮时的处理"""
        if not self.filename:
            QMessageBox.warning(self, "警告", "请先选择视频文件")
            return
            
        # 计算文件名基础部分
        filename_base = os.path.basename(self.filename).split('.')[0]
        
        # 启动处理线程
        self.start_process(
            "colors", 
            imgpath=filename_base, 
            colors_count=colors_count
        )
    
    def on_objects_clicked(self):
        """当用户点击物体检测按钮时的处理"""
        if not self.filename:
            QMessageBox.warning(self, "警告", "请先选择视频文件")
            return
            
        # 计算文件名基础部分
        filename_base = os.path.basename(self.filename).split('.')[0]
        
        # 启动处理线程
        self.start_process(
            "objects", 
            imgpath=filename_base
        )
    
    def on_subtitle_clicked(self, subtitle_value):
        """当用户点击字幕识别按钮时的处理"""
        if not self.filename:
            QMessageBox.warning(self, "警告", "请先选择视频文件")
            return
            
        # 计算文件名基础部分
        filename_base = os.path.basename(self.filename).split('.')[0]
        
        # 启动处理线程
        self.start_process(
            "subtitles", 
            filename=self.filename, 
            imgpath=filename_base, 
            subtitle_value=subtitle_value
        )
    
    def on_shotscale_clicked(self):
        """当用户点击镜头尺度按钮时的处理"""
        if not self.filename:
            QMessageBox.warning(self, "警告", "请先选择视频文件")
            return
            
        # 计算路径
        filename_base = os.path.basename(self.filename).split('.')[0]
        image_save = f"img/{filename_base}/"
        frame_save = f"img/{filename_base}/frame/"
        
        # 启动处理线程
        self.start_process(
            "shotscale", 
            image_save=image_save, 
            frame_save=frame_save
        )

def main():
    # 抑制TensorFlow警告信息
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=全部信息, 1=不显示INFO, 2=不显示INFO和WARNING, 3=只显示ERROR

    app = QApplication(sys.argv)
    # 应用样式
    app.setStyle(QStyleFactory.create("Fusion"))
    
    # 创建启动界面
    splash = Splash()
    splash.show()
    
    # 处理应用事件以确保启动界面显示
    app.processEvents()
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    # 关闭启动界面
    splash.finish(window)
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
