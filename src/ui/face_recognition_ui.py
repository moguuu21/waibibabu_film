import os
from pathlib import Path
from PySide6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QFileDialog, QInputDialog, QMessageBox, QTabWidget,
    QScrollArea, QSplitter, QFrame, QProgressBar, QComboBox, QGridLayout
)
from PySide6.QtGui import QPixmap, QIcon, QImage
from PySide6.QtCore import Qt, Signal, QSize, QThread
import cv2
from datetime import datetime
import numpy as np
from algorithms.face_recognition_module import FaceRecognition
import time


# 添加人脸提取线程类
class FaceExtractionThread(QThread):
    # 定义信号
    progress_signal = Signal(int)  # 进度信号
    finished_signal = Signal(object, object)  # 完成信号，参数为提取的目录和人脸结果
    error_signal = Signal(str)  # 错误信号

    def __init__(self, face_recognition, frames_dir):
        super().__init__()
        self.face_recognition = face_recognition
        self.frames_dir = frames_dir

    def run(self):
        try:
            # 获取帧目录中的文件总数，用于计算进度
            frame_files = [f for f in os.listdir(self.frames_dir)
                           if f.endswith(('.jpg', '.png', '.jpeg'))]
            total_files = len(frame_files)

            if total_files == 0:
                self.error_signal.emit("未在关键帧目录中找到图像文件")
                return

            # 创建输出目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(os.path.dirname(self.frames_dir), f"faces_{timestamp}")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            result_faces = []

            # 处理每个关键帧
            for i, filename in enumerate(frame_files):
                if self.isInterruptionRequested():
                    self.error_signal.emit("操作被用户取消")
                    return

                frame_path = os.path.join(self.frames_dir, filename)

                try:
                    # 加载图片 - 直接调用face_recognition库
                    import face_recognition
                    from PIL import Image

                    image = face_recognition.load_image_file(frame_path)

                    # 检测人脸
                    face_locations = face_recognition.face_locations(image)

                    # 如果存在人脸
                    if face_locations:
                        # 为每个人脸创建条目并保存裁剪图像
                        for j, face_location in enumerate(face_locations):
                            top, right, bottom, left = face_location
                            face_image = image[top:bottom, left:right]
                            pil_image = Image.fromarray(face_image)

                            # 保存裁剪的人脸
                            face_filename = f"{os.path.splitext(filename)[0]}_face{j + 1}.jpg"
                            face_path = os.path.join(output_dir, face_filename)
                            pil_image.save(face_path)

                            # 获取人脸编码进行识别
                            face_encoding = face_recognition.face_encodings(image, [face_location])[0]

                            # 与现有人脸比较
                            name = "未知人物"
                            confidence = "N/A"

                            if len(self.face_recognition.known_face_encodings) > 0:
                                matches = face_recognition.compare_faces(
                                    self.face_recognition.known_face_encodings, face_encoding)
                                face_distances = face_recognition.face_distance(
                                    self.face_recognition.known_face_encodings, face_encoding)
                                best_match_index = np.argmin(face_distances)

                                # 计算置信度
                                distance = face_distances[best_match_index]
                                confidence = round((1 - distance) * 100, 2)

                                if matches[best_match_index]:
                                    name = self.face_recognition.known_face_names[best_match_index]

                            # 添加到结果
                            result_faces.append({
                                "frame": filename,
                                "face_file": face_filename,
                                "face_path": face_path,
                                "name": name,
                                "confidence": confidence,
                                "location": face_location
                            })

                except Exception as e:
                    print(f"处理帧 {filename} 时出错: {str(e)}")

                # 发送进度信号
                progress = int((i + 1) / total_files * 100)
                self.progress_signal.emit(progress)

            # 发送完成信号
            self.finished_signal.emit(output_dir, result_faces)

        except Exception as e:
            self.error_signal.emit(f"提取人脸时发生错误: {str(e)}")


class FaceRecognitionUI(QDockWidget):
    def __init__(self, parent):
        super().__init__('人脸识别', parent)
        self.parent = parent
        self.face_recognition = FaceRecognition()
        self.current_frame_dir = None
        self.extracted_faces_dir = None
        self.face_results = []
        self.selected_face_path = None
        self.compare_face_path = None
        self.extraction_thread = None  # 添加线程对象引用

        self.init_ui()

        # 连接信号
        self.parent.shot_finished.connect(self.on_shot_finished)

    def init_ui(self):
        # 创建主widget和布局
        self.main_widget = QWidget()
        main_layout = QVBoxLayout(self.main_widget)

        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background: #F0F0F0;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #BBDEFB;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #64B5F6;
            }
        """)

        # 创建内容容器
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # 添加标题
        title_label = QLabel("人脸识别管理")
        title_label.setStyleSheet("""
            font-size: 14px; 
            font-weight: bold; 
            color: #1976D2;
            padding: 5px;
            border-bottom: 2px solid #BBDEFB;
            margin-bottom: 5px;
        """)
        content_layout.addWidget(title_label)

        # 创建选项卡
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #BBDEFB;
                border-radius: 4px;
                background-color: white;
                padding: 5px;
            }
            QTabBar::tab {
                background: #E3F2FD;
                color: #1976D2;
                border: 1px solid #BBDEFB;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 6px 10px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #BBDEFB;
                font-weight: bold;
                border-bottom: none;
            }
            QTabBar::tab:hover {
                background: #90CAF9;
            }
        """)

        # 创建三个选项卡的内容
        self.create_extract_tab()
        self.create_recognize_tab()
        self.create_compare_tab()

        # 添加选项卡到tab widget
        self.tab_widget.addTab(self.extract_tab, "提取人脸")
        self.tab_widget.addTab(self.recognize_tab, "识别人脸")
        self.tab_widget.addTab(self.compare_tab, "人脸对比")

        # 将tab widget添加到内容布局
        content_layout.addWidget(self.tab_widget)

        # 设置滚动区域内容
        scroll_area.setWidget(content_widget)

        # 将滚动区域添加到主布局
        main_layout.addWidget(scroll_area)

        # 设置主widget
        self.setWidget(self.main_widget)
        self.setWindowTitle("人脸识别")

        # 初始化状态
        self.update_status_label("准备就绪")

    def create_extract_tab(self):
        """创建提取人脸选项卡"""
        self.extract_tab = QWidget()
        layout = QVBoxLayout(self.extract_tab)
        layout.setSpacing(10)

        # 添加说明标签
        info_label = QLabel("从视频关键帧中提取人脸")
        info_label.setStyleSheet("""
            font-weight: bold; 
            color: #1976D2;
            padding: 5px;
            background-color: #E3F2FD;
            border-radius: 4px;
        """)
        layout.addWidget(info_label)

        # 添加提取按钮
        extract_button = QPushButton("从当前视频关键帧提取人脸")
        extract_button.setIcon(QIcon("assets/icons/face-scan.png"))
        extract_button.setMinimumHeight(36)
        layout.addWidget(extract_button)
        extract_button.clicked.connect(self.extract_faces_from_current_frames)

        # 进度条
        self.extract_progress = QProgressBar()
        self.extract_progress.setVisible(False)
        self.extract_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #BBDEFB;
                border-radius: 3px;
                text-align: center;
                background-color: #F5F5F5;
            }
            QProgressBar::chunk {
                background-color: #64B5F6;
                border-radius: 2px;
            }
        """)
        layout.addWidget(self.extract_progress)

        # 人脸结果显示
        face_section = QWidget()
        face_layout = QVBoxLayout(face_section)
        face_layout.setContentsMargins(0, 0, 0, 0)

        face_title = QLabel("提取的人脸")
        face_title.setStyleSheet("font-weight: bold; color: #555;")
        face_layout.addWidget(face_title)

        self.faces_list = QListWidget()
        self.faces_list.setIconSize(QSize(100, 100))
        self.faces_list.setViewMode(QListWidget.IconMode)
        self.faces_list.setResizeMode(QListWidget.Adjust)
        self.faces_list.setWrapping(True)
        self.faces_list.setMinimumHeight(200)
        self.faces_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #BBDEFB;
                border-radius: 4px;
                background-color: white;
                padding: 5px;
            }
            QListWidget::item {
                border: 1px solid #E3F2FD;
                border-radius: 4px;
                margin: 3px;
            }
            QListWidget::item:selected {
                background-color: #BBDEFB;
                border: 2px solid #1976D2;
            }
            QListWidget::item:hover {
                background-color: #E3F2FD;
            }
        """)
        self.faces_list.itemClicked.connect(self.on_face_selected)
        face_layout.addWidget(self.faces_list)

        layout.addWidget(face_section)

        # 操作按钮
        buttons_section = QWidget()
        buttons_layout = QHBoxLayout(buttons_section)
        buttons_layout.setContentsMargins(0, 0, 0, 0)

        self.add_to_db_button = QPushButton("添加到人脸库")
        self.add_to_db_button.setIcon(QIcon("assets/icons/add-face.png"))
        self.add_to_db_button.clicked.connect(self.add_selected_face_to_db)
        self.add_to_db_button.setEnabled(False)
        buttons_layout.addWidget(self.add_to_db_button)

        self.set_compare_button = QPushButton("设为对比源")
        self.set_compare_button.setIcon(QIcon("assets/icons/compare.png"))
        self.set_compare_button.clicked.connect(self.set_selected_face_as_compare)
        self.set_compare_button.setEnabled(False)
        buttons_layout.addWidget(self.set_compare_button)

        layout.addWidget(buttons_section)

        # 状态标签
        self.status_label = QLabel("准备就绪")
        self.status_label.setStyleSheet("""
            padding: 5px;
            font-style: italic;
            color: #555;
            background-color: #F5F5F5;
            border-radius: 3px;
            margin-top: 5px;
        """)
        layout.addWidget(self.status_label)

    def create_recognize_tab(self):
        """创建识别人脸选项卡"""
        self.recognize_tab = QWidget()
        layout = QVBoxLayout(self.recognize_tab)

        # 添加说明标签
        info_label = QLabel("识别图像中的人脸")
        info_label.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(info_label)

        # 按钮布局
        buttons_layout = QHBoxLayout()

        # 选择图片按钮
        select_image_button = QPushButton("选择图片")
        select_image_button.clicked.connect(self.select_image_for_recognition)
        buttons_layout.addWidget(select_image_button)

        # 识别当前帧按钮
        recognize_current_button = QPushButton("识别当前帧")
        recognize_current_button.clicked.connect(self.recognize_current_frame)
        buttons_layout.addWidget(recognize_current_button)

        # 管理人脸库按钮
        manage_db_button = QPushButton("管理人脸库")
        manage_db_button.clicked.connect(self.manage_face_database)
        buttons_layout.addWidget(manage_db_button)

        layout.addLayout(buttons_layout)

        # 分割器，左侧显示原图，右侧显示识别结果
        splitter = QSplitter(Qt.Horizontal)

        # 左侧：原图
        left_frame = QFrame()
        left_layout = QVBoxLayout(left_frame)
        left_layout.addWidget(QLabel("原始图像:"))
        self.original_image_label = QLabel()
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setMinimumSize(320, 240)
        self.original_image_label.setFrameShape(QFrame.Box)
        self.original_image_label.setStyleSheet("border: 1px solid #cccccc;")
        left_layout.addWidget(self.original_image_label)

        # 右侧：识别结果
        right_frame = QFrame()
        right_layout = QVBoxLayout(right_frame)
        right_layout.addWidget(QLabel("识别结果:"))
        self.result_image_label = QLabel()
        self.result_image_label.setAlignment(Qt.AlignCenter)
        self.result_image_label.setMinimumSize(320, 240)
        self.result_image_label.setFrameShape(QFrame.Box)
        self.result_image_label.setStyleSheet("border: 1px solid #cccccc;")
        right_layout.addWidget(self.result_image_label)

        # 添加识别信息区域
        self.recognition_info = QLabel()
        self.recognition_info.setWordWrap(True)
        self.recognition_info.setStyleSheet("background-color: #f0f0f0; padding: 5px;")
        right_layout.addWidget(self.recognition_info)

        # 添加到分割器
        splitter.addWidget(left_frame)
        splitter.addWidget(right_frame)

        # 设置分割比例
        splitter.setSizes([400, 400])

        # 添加到主布局
        layout.addWidget(splitter)

    def create_compare_tab(self):
        """创建人脸对比选项卡"""
        self.compare_tab = QWidget()
        layout = QVBoxLayout(self.compare_tab)

        # 添加说明标签
        info_label = QLabel("比较两张人脸图片")
        info_label.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(info_label)

        # 图像选择区域
        images_layout = QHBoxLayout()

        # 左侧图像
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("图像1:"))
        self.image1_label = QLabel()
        self.image1_label.setAlignment(Qt.AlignCenter)
        self.image1_label.setMinimumSize(200, 200)
        self.image1_label.setFrameShape(QFrame.Box)
        self.image1_label.setStyleSheet("border: 1px solid #cccccc;")
        left_layout.addWidget(self.image1_label)

        self.select_image1_button = QPushButton("选择图像1")
        self.select_image1_button.clicked.connect(lambda: self.select_compare_image(1))
        left_layout.addWidget(self.select_image1_button)

        # 右侧图像
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("图像2:"))
        self.image2_label = QLabel()
        self.image2_label.setAlignment(Qt.AlignCenter)
        self.image2_label.setMinimumSize(200, 200)
        self.image2_label.setFrameShape(QFrame.Box)
        self.image2_label.setStyleSheet("border: 1px solid #cccccc;")
        right_layout.addWidget(self.image2_label)

        self.select_image2_button = QPushButton("选择图像2")
        self.select_image2_button.clicked.connect(lambda: self.select_compare_image(2))
        right_layout.addWidget(self.select_image2_button)

        # 添加到水平布局
        images_layout.addLayout(left_layout)
        images_layout.addLayout(right_layout)

        # 添加到主布局
        layout.addLayout(images_layout)

        # 比较按钮
        self.compare_button = QPushButton("比较人脸")
        self.compare_button.clicked.connect(self.compare_selected_faces)
        self.compare_button.setEnabled(False)
        layout.addWidget(self.compare_button)

        # 比较结果区域
        self.comparison_result_label = QLabel()
        self.comparison_result_label.setAlignment(Qt.AlignCenter)
        self.comparison_result_label.setMinimumHeight(300)
        self.comparison_result_label.setFrameShape(QFrame.Box)
        self.comparison_result_label.setStyleSheet("border: 1px solid #cccccc;")
        layout.addWidget(self.comparison_result_label)

        # 比较信息
        self.comparison_info = QLabel()
        self.comparison_info.setWordWrap(True)
        self.comparison_info.setStyleSheet("background-color: #f0f0f0; padding: 5px;")
        layout.addWidget(self.comparison_info)

    def update_status_label(self, message):
        """更新状态标签"""
        self.status_label.setText(message)

    def on_shot_finished(self):
        """当镜头分割完成时调用"""
        # 获取当前视频的帧目录
        if self.parent.filename:
            video_name = os.path.splitext(os.path.basename(self.parent.filename))[0]
            self.current_frame_dir = os.path.join("img", video_name, "frame")
            self.update_status_label(f"视频关键帧目录：{self.current_frame_dir}")

    def extract_faces_from_current_frames(self):
        """从当前视频帧中提取人脸 - 在后台线程中执行"""
        if not self.current_frame_dir or not os.path.exists(self.current_frame_dir):
            QMessageBox.warning(self, "警告", "未找到视频关键帧目录，请先进行视频分镜")
            return

        # 显示进度条
        self.extract_progress.setVisible(True)
        self.extract_progress.setValue(0)

        self.update_status_label("正在提取人脸...")

        # 清空人脸列表
        self.faces_list.clear()

        # 如果有正在运行的线程，停止它
        if self.extraction_thread and self.extraction_thread.isRunning():
            self.extraction_thread.requestInterruption()
            self.extraction_thread.wait()

        # 创建并启动新线程
        self.extraction_thread = FaceExtractionThread(self.face_recognition, self.current_frame_dir)

        # 连接信号
        self.extraction_thread.progress_signal.connect(self.update_extraction_progress)
        self.extraction_thread.finished_signal.connect(self.on_extraction_finished)
        self.extraction_thread.error_signal.connect(self.on_extraction_error)

        # 启动线程
        self.extraction_thread.start()

    def update_extraction_progress(self, progress):
        """更新提取进度"""
        self.extract_progress.setValue(progress)

    def on_extraction_finished(self, output_dir, face_results):
        """当提取完成时调用"""
        self.extracted_faces_dir = output_dir
        self.face_results = face_results

        # 更新UI
        if self.face_results:
            self.load_faces_to_list()
            self.update_status_label(f"成功提取 {len(self.face_results)} 个人脸")
        else:
            self.update_status_label("未在关键帧中找到人脸")

        # 隐藏进度条
        self.extract_progress.setVisible(False)

    def on_extraction_error(self, error_message):
        """当提取出错时调用"""
        self.update_status_label(f"提取人脸出错：{error_message}")
        QMessageBox.critical(self, "错误", f"提取人脸时发生错误：{error_message}")

        # 隐藏进度条
        self.extract_progress.setVisible(False)

    def load_faces_to_list(self):
        """将提取的人脸加载到列表中"""
        self.faces_list.clear()

        for face in self.face_results:
            item = QListWidgetItem()
            pixmap = QPixmap(face["face_path"])
            item.setIcon(QIcon(pixmap))

            # 设置工具提示为人脸信息
            confidence = face["confidence"]
            if confidence != "N/A":
                tooltip = f"名称: {face['name']}\n置信度: {confidence}%\n来源帧: {face['frame']}"
            else:
                tooltip = f"名称: {face['name']}\n来源帧: {face['frame']}"

            item.setToolTip(tooltip)

            # 存储人脸路径作为用户数据
            item.setData(Qt.UserRole, face["face_path"])

            self.faces_list.addItem(item)

    def on_face_selected(self, item):
        """当人脸被选中时调用"""
        # 获取选中的人脸路径
        self.selected_face_path = item.data(Qt.UserRole)

        # 启用相关按钮
        self.add_to_db_button.setEnabled(True)
        self.set_compare_button.setEnabled(True)

    def add_selected_face_to_db(self):
        """将选中的人脸添加到数据库"""
        if not self.selected_face_path:
            return

        # 获取人名
        name, ok = QInputDialog.getText(self, "添加人脸", "请输入此人的姓名:")

        if ok and name:
            # 添加到数据库
            success, message = self.face_recognition.add_face(self.selected_face_path, name)

            if success:
                QMessageBox.information(self, "成功", message)
                self.update_status_label(message)
            else:
                QMessageBox.warning(self, "警告", message)
                self.update_status_label(message)

    def set_selected_face_as_compare(self):
        """将选中的人脸设为对比源"""
        if not self.selected_face_path:
            return

        # 设置为对比源
        self.compare_face_path = self.selected_face_path

        # 更新对比选项卡中的图像
        pixmap = QPixmap(self.selected_face_path)
        self.image1_label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio))

        # 检查是否可以进行对比
        self.check_compare_button_state()

        # 切换到对比选项卡
        self.tab_widget.setCurrentIndex(2)  # 对比选项卡索引为2

        self.update_status_label(f"已设置对比源：{os.path.basename(self.selected_face_path)}")

    def select_image_for_recognition(self):
        """选择图像进行识别"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像", "", "图像文件 (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            # 显示原始图像
            pixmap = QPixmap(file_path)
            self.original_image_label.setPixmap(pixmap.scaled(320, 240, Qt.KeepAspectRatio))

            # 执行人脸识别
            self.recognize_image(file_path)

    def recognize_current_frame(self):
        """识别当前播放帧中的人脸"""
        try:
            # 检查主窗口对象是否存在
            if not hasattr(self, 'parent') or not self.parent:
                QMessageBox.warning(self, "警告", "无法访问主窗口")
                return

            # 检查播放器对象是否存在
            if not hasattr(self.parent, 'player') or not self.parent.player:
                QMessageBox.warning(self, "警告", "播放器未初始化")
                return

            # 检查是否有视频播放
            if not self.parent.filename or not os.path.exists(self.parent.filename):
                QMessageBox.warning(self, "警告", "请先加载视频文件")
                return

            # 使用OpenCV从视频文件截取当前帧
            try:
                # 获取当前播放时间(毫秒)
                current_time_ms = self.parent.player.mediaplayer.get_time()

                if current_time_ms < 0:
                    QMessageBox.warning(self, "警告", "无法获取当前播放时间，请确保视频正在播放")
                    return

                # 打开视频文件
                cap = cv2.VideoCapture(self.parent.filename)
                if not cap.isOpened():
                    QMessageBox.warning(self, "警告", "无法打开视频文件")
                    return

                # 计算当前帧的位置
                fps = cap.get(cv2.CAP_PROP_FPS)
                current_frame = int(current_time_ms / 1000.0 * fps)

                # 设置到指定帧
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

                # 读取当前帧
                ret, frame = cap.read()
                cap.release()

                if not ret:
                    QMessageBox.warning(self, "警告", "无法读取当前帧")
                    return

                # 保存当前帧到临时文件
                temp_dir = os.path.join("img", "temp")
                os.makedirs(temp_dir, exist_ok=True)
                temp_file = os.path.join(temp_dir, f"current_frame_{int(time.time())}.jpg")
                cv2.imwrite(temp_file, frame)

                # 显示当前帧
                pixmap = QPixmap(temp_file)
                self.original_image_label.setPixmap(pixmap.scaled(320, 240, Qt.KeepAspectRatio))

                # 执行人脸识别
                self.recognize_image(temp_file)

            except Exception as e:
                QMessageBox.warning(self, "错误", f"处理当前帧时出错: {str(e)}")
                return

        except Exception as e:
            QMessageBox.critical(self, "错误", f"识别当前帧时出错: {str(e)}")
            print(f"识别当前帧错误: {str(e)}")

    def recognize_image(self, image_path):
        """识别图像中的人脸"""
        try:
            # 执行人脸识别
            result_image, face_results = self.face_recognition.recognize_faces(image_path)

            # 将OpenCV图像转换为Qt图像
            height, width, channel = result_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(result_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

            # 显示结果图像
            pixmap = QPixmap.fromImage(q_image)
            self.result_image_label.setPixmap(pixmap.scaled(320, 240, Qt.KeepAspectRatio))

            # 显示识别信息
            if face_results:
                info_text = f"检测到 {len(face_results)} 个人脸:\n"
                for i, face in enumerate(face_results):
                    confidence = face["confidence"]
                    if confidence != "N/A":
                        info_text += f"{i + 1}. {face['name']} (置信度: {confidence}%)\n"
                    else:
                        info_text += f"{i + 1}. {face['name']}\n"

                    # 添加属性显示
                    if face.get("attributes"):
                        info_text += "   属性: "
                        attrs = [f"{name}({prob})" for name, prob in face["attributes"]]
                        info_text += ", ".join(attrs) + "\n"

                self.recognition_info.setText(info_text)
            else:
                self.recognition_info.setText("未检测到人脸")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"识别人脸时发生错误：{str(e)}")
            self.recognition_info.setText(f"错误：{str(e)}")


    def manage_face_database(self):
        """管理人脸数据库"""
        # 打开人脸数据库目录
        if os.path.exists(self.face_recognition.face_db_dir):
            os.startfile(os.path.abspath(self.face_recognition.face_db_dir))
        else:
            QMessageBox.warning(self, "警告", "人脸数据库目录不存在")

    def select_compare_image(self, image_num):
        """选择用于比较的图像"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"选择图像{image_num}", "", "图像文件 (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            # 显示选择的图像
            pixmap = QPixmap(file_path)

            if image_num == 1:
                self.image1_label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio))
                self.compare_face_path = file_path
            else:
                self.image2_label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio))
                self.selected_face_path = file_path

            # 检查是否可以进行比较
            self.check_compare_button_state()

    def check_compare_button_state(self):
        """检查是否可以进行人脸比较"""
        if self.compare_face_path and self.selected_face_path:
            self.compare_button.setEnabled(True)
        else:
            self.compare_button.setEnabled(False)

    def compare_selected_faces(self):
        """比较选择的两张人脸图片"""
        if not self.compare_face_path or not self.selected_face_path:
            return

        try:
            # 创建比较图像
            comparison_image_path = self.face_recognition.create_face_comparison_image(
                self.compare_face_path, self.selected_face_path
            )

            # 显示比较结果
            pixmap = QPixmap(comparison_image_path)
            self.comparison_result_label.setPixmap(pixmap.scaled(800, 300, Qt.KeepAspectRatio))

            # 获取比较详情
            is_match, message, similarity = self.face_recognition.compare_face(
                self.compare_face_path, self.selected_face_path
            )

            # 显示比较信息
            if is_match:
                info_text = f"<span style='color:green;'>✓ {message}</span><br>相似度: {similarity}%"
            else:
                info_text = f"<span style='color:red;'>✗ {message}</span><br>相似度: {similarity}%"

            self.comparison_info.setText(info_text)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"比较人脸时发生错误：{str(e)}")
            self.comparison_info.setText(f"错误：{str(e)}")