import os
import re
import logging
import easyocr
import cv2
import csv
from algorithms.wordcloud2frame import WordCloud2Frame

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SubtitleEasyOCR")

class SubtitleProcessor:
    def __init__(self):
        # 设置EasyOCR模型路径到项目目录下的models/EasyOCR
        model_storage_directory = os.path.abspath(
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        "..", "models", "EasyOCR")
        )
        
        # 确保目录存在
        os.makedirs(model_storage_directory, exist_ok=True)
        logger.info(f"使用EasyOCR模型目录: {model_storage_directory}")
        
        # 初始化EasyOCR reader，使用指定的模型目录
        self.reader = easyocr.Reader(
            ['ch_sim', 'en'], 
            model_storage_directory=model_storage_directory,
            download_enabled=True  # 如果模型不存在，允许下载
        )
        logger.info("EasyOCR初始化完成")
        
    def getsubtitleEasyOcr(self,v_path,save_path,subtitleValue):
        logger.info(f"开始字幕识别，视频路径: {v_path}, 保存路径: {save_path}")
        path=v_path
        cap = cv2.VideoCapture(path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        subtitleList = []
        subtitleStr = ""
        i = 0
        _, frame = cap.read(i)
        h,w=frame.shape[0:2]#图片尺寸，截取下三分之一和中间五分之四作为字幕检测区域
        start_h = (h // 3)*2
        end_h = h
        start_w = w // 20
        end_w = (w // 20) * 19
        img1=frame[start_h:end_h,start_w:end_w,:]
        i=i+1
        th=0.2
        while i<frame_count:
            if img1 is None:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            _, frame = cap.read(i)
            h, w = frame.shape[0:2]  # 图片尺寸，截取下五分之一和中间五分之四作为字幕检测区域
            start_h = (h // 5)*4
            end_h = h
            start_w = w // 10
            end_w = (w // 10) * 9
            img2 = frame[start_h:end_h, start_w:end_w]
            subtitle_event= self.subtitleDetect(img1, img2, th)
            if subtitle_event:
                wordslist = self.reader.readtext(img2)
                # print("wordlist",wordslist)
                # subtitleStr=subtitleStr+
                for w in wordslist:
                    # print('w',w,w[1])
                    if w[1] is not None:
                        # 打印检测到的文字，用于调试
                        logger.info(f"检测到文字: {w[1]}")
                        
                        # 去除不想要的英文或数字序列
                        if (not subtitleList or w[1] != subtitleList[-1][1]) and self.is_valid_subtitle(w[1]):
                            subtitleList.append([i,w[1]])
                            subtitleStr=subtitleStr+w[1]+'\n'
            else:
                img1=img2
            i = i + subtitleValue
            #12-120，默认48帧
        cap.release()
        
        # 打印检测到的字幕数量
        logger.info(f"共检测到 {len(subtitleList)} 条字幕")
        
        # 生成词云
        wc2f=WordCloud2Frame()
        tf = wc2f.wordfrequencyStr(subtitleStr)
        wc2f.plotwordcloud(tf,save_path,"subtitle")

        return subtitleStr,subtitleList

    def subtitle2Srt(self,subtitleList, savePath):
        # path为输出路径和文件名，newline=''是为了不出现空行
        csvpath=savePath+"subtitle.csv"
        srtFile=savePath+"subtitle.srt"
        
        # 保存CSV文件
        try:
            with open(csvpath, "w", newline='', encoding='utf-8') as csvFile:
                # name为列名
                name = ['FrameId','Subtitles']
                writer = csv.writer(csvFile)
                writer.writerow(name)
                for i in range(len(subtitleList)):
                    datarow=[subtitleList[i][0]]
                    datarow.append(subtitleList[i][1])
                    writer.writerow(datarow)
                logger.info(f"已保存字幕CSV文件: {csvpath}")
        except Exception as e:
            logger.error(f"保存CSV文件失败: {str(e)}")
            
        # 保存SRT文件
        try:
            with open(srtFile, 'w', encoding='utf-8') as f:
                for i in range(len(subtitleList)):
                    # 写入字幕编号
                    f.write(f"{i+1}\n")
                    
                    # 计算时间码
                    frame_id = subtitleList[i][0]
                    start_time = self._frame_to_timecode(frame_id, 25)  # 假设帧率为25fps
                    end_time = self._frame_to_timecode(frame_id + 25, 25)  # 显示1秒
                    
                    # 写入时间码
                    f.write(f"{start_time} --> {end_time}\n")
                    
                    # 写入字幕文本
                    f.write(f"{subtitleList[i][1]}\n\n")
                    
                logger.info(f"已保存字幕SRT文件: {srtFile}")
        except Exception as e:
            logger.error(f"保存SRT文件失败: {str(e)}")
            
    def _frame_to_timecode(self, frame_id, fps):
        """将帧号转换为SRT时间码格式 (00:00:00,000)"""
        total_seconds = frame_id / fps
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int((total_seconds - int(total_seconds)) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def contains_english(self,text):
        # 使用正则表达式匹配英文字符
        english_pattern = re.compile(r'[a-zA-Z]')
        return bool(english_pattern.search(text))

    def is_valid_subtitle(self, text):
        """检查字幕文本是否有效"""
        # 如果文本太短，可能不是有效字幕
        if len(text) < 2:
            return False
            
        # 如果文本全是数字或标点符号，可能不是有效字幕
        if re.match(r'^[\d\s\.\,\?\!\;\:\(\)\[\]\{\}\-\+\=\_\*\&\^\%\$\#\@\~\`\'\"]+$', text):
            return False
            
        # 如果是单个英文字母加数字可能是误识别
        if re.match(r'^[a-zA-Z]\d+$', text):
            return False
            
        # 检查文本是否包含中文字符
        contains_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
        contains_english = bool(re.search(r'[a-zA-Z]', text))
        
        # 如果只包含少量英文字符且没有中文，可能是误识别
        if contains_english and not contains_chinese and len(text) <= 3:
            return False
            
        return True

    def cmpHash(self,hash1, hash2):
        n = 0
        # hash长度不同则返回-1代表传参出错
        if len(hash1) != len(hash2):
            return -1
        # 遍历判断
        for i in range(len(hash1)):
            # 不相等则n计数+1，n最终为相似度
            if hash1[i] != hash2[i]:
                n = n + 1
        n = n/len(hash1)
        return n

    def aHash(self, img):
        if img is None:
            print("none")
        imgsmall = cv2.resize(img, (16, 4))
        # 转换为灰度图
        gray = cv2.cvtColor(imgsmall, cv2.COLOR_BGR2GRAY)
        # s为像素和初值为0，hash_str为hash值初值为''
        s = 0
        hash_str = ''
        # 遍历累加求像素和
        for i in range(4):
            for j in range(16):
                s = s + gray[i, j]
        # 求平均灰度
        avg = s / 64
        # 遍历图像的每个像素，并比较每个像素的灰度值是否大于平均灰度值 avg。如果大于 avg，则将 '1' 添加到 hash_str 中，否则添加 '0'。这样就生成了一个二进制的哈希字符串
        for i in range(4):
            for j in range(16):
                if gray[i, j] > avg:
                    hash_str = hash_str + '1'
                else:
                    hash_str = hash_str + '0'
        return hash_str

    def subtitleDetect(self,img1, img2, th):
        hash1 = self.aHash(img1)
        hash2 = self.aHash(img2)
        n = self.cmpHash(hash1, hash2)  # 不同加1，相同为0
        if n > th:
            subtitle_event=True
        else:
            subtitle_event=False
        return subtitle_event

