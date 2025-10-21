#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
pyCinemetrics 模型文件下载工具
该脚本用于下载pyCinemetrics所需的模型文件
"""

import os
import sys
import logging
import platform
import requests
import shutil
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ModelDownloader")

# 模型文件配置
MODELS = {
    "yolov3.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
    "yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights",
    "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
    "shotscale_resnet.pth": "https://huggingface.co/datasets/jinyuandev/models/resolve/main/shotscale_resnet.pth"
}

def download_file(url, destination):
    """从URL下载文件到指定路径，显示进度条
    
    Args:
        url: 下载链接
        destination: 目标保存路径
    
    Returns:
        bool: 是否成功下载
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 如果响应状态不是200，将引发HTTPError异常
        
        # 获取文件大小
        file_size = int(response.headers.get('content-length', 0))
        
        # 创建进度条
        desc = os.path.basename(destination)
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=desc) as pbar:
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        return True
    except Exception as e:
        logger.error(f"下载 {url} 时出错: {str(e)}")
        if os.path.exists(destination):
            os.remove(destination)  # 删除可能不完整的文件
        return False

def ensure_directory_exists(directory):
    """确保目录存在，不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"已创建目录: {directory}")

def download_model_files():
    """下载所有必要的模型文件"""
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 模型保存目录
    models_dir = os.path.abspath(os.path.join(script_dir, "..", "models"))
    
    # 创建模型目录（如果不存在）
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
        logger.info(f"创建模型目录: {models_dir}")
    
    logger.info(f"模型将被下载到: {models_dir}")
    
    # 检查哪些模型需要下载
    models_to_download = []
    for model_name, url in MODELS.items():
        model_path = os.path.join(models_dir, model_name)
        if not os.path.exists(model_path):
            models_to_download.append((model_name, url, model_path))
    
    if not models_to_download:
        logger.info("所有模型文件都已存在，无需下载")
        return
    
    # 开始下载缺失的模型
    logger.info(f"需要下载 {len(models_to_download)} 个模型文件")
    
    success_count = 0
    for model_name, url, model_path in models_to_download:
        logger.info(f"正在下载 {model_name}...")
        if download_file(url, model_path):
            logger.info(f"{model_name} 下载成功")
            success_count += 1
        else:
            logger.error(f"{model_name} 下载失败")
    
    # 下载完成，显示结果
    if success_count == len(models_to_download):
        logger.info("所有模型文件下载成功！")
    else:
        logger.warning(f"部分模型下载失败: {success_count}/{len(models_to_download)} 成功")
    
    # 特别提示
    logger.info("\n其他依赖项:")
    logger.info("1. 如需使用字幕识别功能，请确保已安装Tesseract OCR")
    logger.info("   Windows: https://github.com/UB-Mannheim/tesseract/wiki")
    logger.info("   Linux: sudo apt-get install tesseract-ocr")
    logger.info("   macOS: brew install tesseract")
    logger.info("2. 请确保Tesseract OCR已添加到系统PATH")

def check_model_files():
    """检查模型文件是否存在
    
    Returns:
        tuple: (模型目录, 存在的文件列表, 缺失的文件列表, 文件大小字典)
    """
    # 获取模型目录路径
    models_dir = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "..", "models"
    ))
    
    # 如果目录不存在，创建目录
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        logger.info(f"创建模型目录: {models_dir}")
    
    # 检查EasyOCR模型目录
    easyocr_dir = os.path.join(models_dir, "EasyOCR")
    if not os.path.exists(easyocr_dir):
        os.makedirs(easyocr_dir)
        logger.info(f"创建EasyOCR模型目录: {easyocr_dir}")
    
    # 需要检查的文件列表
    required_files = [
        "yolov3.cfg",          # YOLO配置文件
        "yolov3.weights",      # YOLO权重文件
        "coco.names",          # COCO数据集类别名称
        "shotscale_resnet.pth" # 镜头尺度识别模型
    ]
    
    # 检查文件是否存在
    existing_files = []
    missing_files = []
    file_sizes = {}
    
    for file_name in required_files:
        file_path = os.path.join(models_dir, file_name)
        if os.path.exists(file_path):
            existing_files.append(file_name)
            # 获取文件大小
            size_bytes = os.path.getsize(file_path)
            if size_bytes < 1024:
                size_str = f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                size_str = f"{size_bytes/1024:.1f} KB"
            else:
                size_str = f"{size_bytes/(1024*1024):.1f} MB"
            file_sizes[file_name] = size_str
        else:
            missing_files.append(file_name)
    
    return models_dir, existing_files, missing_files, file_sizes

def main():
    """主函数"""
    print("=" * 50)
    print(" PyCinemetrics 模型文件检查工具 ")
    print("=" * 50)
    print()
    
    try:
        # 检查模型文件
        models_dir, existing_files, missing_files, file_sizes = check_model_files()
        
        print(f"模型目录: {models_dir}")
        print()
        
        if existing_files:
            print("已存在的模型文件:")
            for file_name in existing_files:
                print(f"  - {file_name} ({file_sizes[file_name]})")
        else:
            print("未找到任何模型文件")
        
        print()
        
        if missing_files:
            print("缺失的模型文件:")
            for file_name in missing_files:
                print(f"  - {file_name}")
            print()
            print("请下载缺失的模型文件并放置在模型目录中。")
            print("您可以从项目仓库或官方网站获取这些文件。")
        else:
            print("所有必需的模型文件都已存在。")
        
        print()
        print("EasyOCR信息:")
        print("字幕识别功能使用EasyOCR，首次运行时会自动下载语言模型。")
        print("模型将保存在以下目录: " + os.path.join(models_dir, "EasyOCR"))
        print()
        print("如遇到问题，请参考项目文档或联系开发者获取帮助。")
    
    except Exception as e:
        logger.error(f"检查模型文件时出错: {str(e)}")
        print(f"发生错误: {str(e)}")
    
    print()
    input("按Enter键退出...")

if __name__ == "__main__":
    try:
        download_model_files()
        input("\n按回车键退出...")
    except KeyboardInterrupt:
        logger.info("用户中断下载")
        sys.exit(1)
    except Exception as e:
        logger.error(f"发生错误: {str(e)}")
        sys.exit(1) 