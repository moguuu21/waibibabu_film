const API = {
    base: '',
    endpoint(path) { return `${this.base}${path}`; }
};

const dom = {
    video: document.getElementById('main_player'),
    dropzone: document.querySelector('.video-container'),
    videoPathInput: document.getElementById('video_path'),
    fileInput: document.getElementById('file_input'),
    status: document.getElementById('status_log'),
    badge: document.getElementById('shot_count_badge'),
    
    // Result views
    tabs: document.querySelectorAll('.tab-btn'),
    views: document.querySelectorAll('.result-view'),
    
    timeline: document.getElementById('timeline_track'),
    subList: document.getElementById('subtitle_list'),
    chartBox: document.getElementById('chart_container')
};

// 状态管理
let currentVideoData = {
    duration: 0,
    totalFrames: 0
};

// --------------- 工具函数 ---------------
function log(msg) {
    const time = new Date().toLocaleTimeString();
    dom.status.textContent = `[${time}] ${msg}`;
    // 自动滚动到底部
    dom.status.scrollTop = dom.status.scrollHeight;
}

async function postJSON(url, payload) {
    const res = await fetch(url, { 
        method: 'POST', 
        headers: { 'Content-Type': 'application/json' }, 
        body: JSON.stringify(payload) 
    });
    // 尝试解析 JSON，如果失败则抛出文本错误
    const text = await res.text();
    let json;
    try {
        json = JSON.parse(text);
    } catch (e) {
        throw new Error(`Server returned invalid JSON: ${text.substring(0, 100)}...`);
    }
    
    if (!res.ok || !json.ok) {
        throw new Error(json.error || res.statusText);
    }
    return json;
}

// 格式化秒数为 MM:SS
function formatTime(seconds) {
    if (!seconds && seconds !== 0) return "--:--";
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

// --------------- Tab 切换 ---------------
dom.tabs.forEach(btn => {
    btn.addEventListener('click', () => {
        dom.tabs.forEach(b => b.classList.remove('active'));
        dom.views.forEach(v => v.classList.remove('active'));
        
        btn.classList.add('active');
        const targetId = btn.getAttribute('data-target');
        document.getElementById(targetId).classList.add('active');
    });
});

function switchToTab(targetId) {
    const btn = document.querySelector(`.tab-btn[data-target="${targetId}"]`);
    if(btn) btn.click();
}

// --------------- 视频加载逻辑 ---------------
async function handleFileUpload(file) {
    if (!file) return;
    
    log('正在上传视频...');
    const fd = new FormData();
    fd.append('file', file);
    
    try {
        const res = await fetch(API.endpoint('/api/upload'), { method: 'POST', body: fd });
        const json = await res.json();
        
        if (json.ok) {
            log(`上传成功: ${json.data.filename}`);
            
            dom.videoPathInput.value = json.data.saved_path;
            
            // 设置预览源
            const filename = json.data.filename;
            dom.video.src = `/uploads/${filename}`;
            dom.video.style.display = 'block';
            document.getElementById('video_placeholder').style.display = 'none';
            
            // 重置状态
            dom.timeline.innerHTML = '<div class="placeholder-text">请在右侧运行“镜头切分”</div>';
            dom.subList.innerHTML = '<div class="placeholder-text">请在右侧运行“字幕识别”</div>';
            dom.chartBox.innerHTML = '<div class="placeholder-text">暂无图表</div>';
            
            dom.video.onloadedmetadata = () => {
                currentVideoData.duration = dom.video.duration;
                log(`视频加载完成: ${currentVideoData.duration.toFixed(2)}s`);
            };
        } else {
            throw new Error(json.error);
        }
    } catch (err) {
        log(`上传失败: ${err.message}`);
    }
}

// --------------- 拖拽事件 ---------------
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dom.dropzone.addEventListener(eventName, (e) => {
        e.preventDefault(); 
        e.stopPropagation();
    }, false);
});

dom.dropzone.addEventListener('dragover', () => dom.dropzone.classList.add('dragover'));
dom.dropzone.addEventListener('dragleave', () => dom.dropzone.classList.remove('dragover'));
dom.dropzone.addEventListener('drop', (e) => {
    dom.dropzone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length) handleFileUpload(files[0]);
});

document.getElementById('btn_select_file').addEventListener('click', () => dom.fileInput.click());
dom.fileInput.addEventListener('change', (e) => handleFileUpload(e.target.files[0]));


// --------------- 1. 镜头切分 ---------------
document.getElementById('btn_shotcut').addEventListener('click', async () => {
    const video_path = dom.videoPathInput.value;
    if (!video_path) { log('请先上传视频'); return; }

    const th = parseFloat(document.getElementById('th').value);
    
    log('开始镜头切分 (TransNetV2)... 这可能需要一段时间');
    switchToTab('view_timeline');
    dom.timeline.innerHTML = '<div class="placeholder-text"><i class="fas fa-spinner fa-spin"></i> 分析中...</div>';
    
    try {
        const data = await postJSON(API.endpoint('/api/shotcut'), { video_path, th });
        log(data.message);
        
        const scenes = data.data && data.data.scenes;
        if (scenes && scenes.length > 0) {
            renderTimeline(scenes);
        } else {
            dom.timeline.innerHTML = '<div class="placeholder-text">未检测到镜头切换</div>';
        }
    } catch (err) {
        log(`切分失败: ${err.message}`);
        dom.timeline.innerHTML = '<div class="placeholder-text error">分析出错</div>';
    }
});

function renderTimeline(scenes) {
    dom.timeline.innerHTML = ''; 
    dom.badge.textContent = scenes.length;
    
    // 获取最后一帧作为总帧数估算
    const lastScene = scenes[scenes.length - 1];
    // 兼容字典或数组格式
    const totalFrames = (Array.isArray(lastScene)) ? lastScene[1] : lastScene.end_frame;
    
    scenes.forEach((shot, index) => {
        let start, end;
        if (Array.isArray(shot)) {
            start = shot[0]; end = shot[1];
        } else {
            start = shot.start_frame; end = shot.end_frame;
        }
        
        const el = document.createElement('div');
        el.className = 'shot-block';
        
        const durationFrames = end - start;
        const widthPercent = (durationFrames / totalFrames) * 100;
        
        el.style.width = `calc(${widthPercent}% - 1px)`;
        if (widthPercent < 0.5) el.style.minWidth = '2px';
        
        el.style.opacity = (index % 2 === 0) ? 1 : 0.85;

        // Tooltip
        const tooltip = document.createElement('div');
        tooltip.className = 'shot-tooltip';
        
        const startTime = (start / totalFrames) * currentVideoData.duration;
        const durationSec = (durationFrames / totalFrames) * currentVideoData.duration;
        
        tooltip.innerText = `Shot ${index + 1}\n${formatTime(startTime)}\nDur: ${durationSec.toFixed(1)}s`;
        el.appendChild(tooltip);
        
        el.addEventListener('click', () => {
            document.querySelectorAll('.shot-block').forEach(b => b.classList.remove('active'));
            el.classList.add('active');
            if (currentVideoData.duration) {
                dom.video.currentTime = startTime;
                dom.video.play();
            }
        });

        dom.timeline.appendChild(el);
    });
}


// --------------- 2. 字幕识别 ---------------
document.getElementById('btn_subtitles').addEventListener('click', async () => {
    const video_path = dom.videoPathInput.value;
    if (!video_path) { log('请先上传视频'); return; }

    const val = parseInt(document.getElementById('subtitle_value').value) || 48;
    log('开始字幕识别... (请耐心等待)');
    
    switchToTab('view_subtitles');
    dom.subList.innerHTML = '<div class="placeholder-text"><i class="fas fa-spinner fa-spin"></i> 识别中...</div>';

    try {
        const response = await postJSON(API.endpoint('/api/subtitles'), { video_path, subtitle_value: val });
        console.log("Subtitle response:", response);
        log(response.message);
        
        if (response.data && response.data.srt_content && response.data.srt_content.trim() !== "") {
            renderSubtitles(response.data.srt_content);
        } else {
            dom.subList.innerHTML = '<div class="placeholder-text">识别结果为空</div>';
        }
    } catch (err) {
        log('错误: ' + err.message);
        dom.subList.innerHTML = `<div class="placeholder-text error">识别失败: ${err.message}</div>`;
    }
});

function renderSubtitles(srtContent) {
    dom.subList.innerHTML = '';
    const normalizedContent = srtContent.replace(/\r\n/g, '\n');
    const blocks = normalizedContent.split(/\n{2,}/);
    let validCount = 0;

    blocks.forEach(block => {
        const lines = block.trim().split('\n');
        if (lines.length >= 2) {
            const timeLineIndex = lines.findIndex(line => line.includes('-->'));
            if (timeLineIndex !== -1) {
                const timeLine = lines[timeLineIndex].trim();
                const text = lines.slice(timeLineIndex + 1).join(' ').trim();
                
                if (!text) return;

                // Parse 00:00:01,000
                const timeMatch = timeLine.match(/(\d{2}):(\d{2}):(\d{2})[,.](\d{3})/);
                let startTime = 0;
                if (timeMatch) {
                    startTime = parseInt(timeMatch[1]) * 3600 + 
                                parseInt(timeMatch[2]) * 60 + 
                                parseInt(timeMatch[3]) + 
                                parseInt(timeMatch[4]) / 1000;
                }

                const el = document.createElement('div');
                el.className = 'sub-item';
                el.innerHTML = `
                    <span class="sub-time"><i class="far fa-clock"></i> ${timeLine}</span>
                    <span class="sub-text">${text}</span>
                `;
                
                el.addEventListener('click', () => {
                    document.querySelectorAll('.sub-item').forEach(i => i.classList.remove('active'));
                    el.classList.add('active');
                    dom.video.currentTime = startTime;
                    dom.video.play();
                });
                
                dom.subList.appendChild(el);
                validCount++;
            }
        }
    });

    if (validCount === 0) {
         dom.subList.innerHTML = '<div class="placeholder-text">无法解析 SRT 内容</div>';
    }
}


// --------------- 3. 图表分析 (色彩/物体/景别) ---------------
async function runVisualTask(apiUrl, payload, title) {
    const video_path = dom.videoPathInput.value;
    if (!video_path) { log('请先上传视频'); return; }

    log(`正在进行${title}...`);
    switchToTab('view_charts');
    dom.chartBox.innerHTML = '<div class="placeholder-text"><i class="fas fa-spinner fa-spin"></i> 处理中...</div>';

    try {
        const data = await postJSON(API.endpoint(apiUrl), { video_path, ...payload });
        log(data.message);
        renderCharts(data.results.files);
    } catch (err) {
        log('错误: ' + err.message);
        dom.chartBox.innerHTML = '<div class="placeholder-text error">处理失败</div>';
    }
}

document.getElementById('btn_colors').addEventListener('click', () => {
    const count = parseInt(document.getElementById('colors_count').value) || 5;
    runVisualTask('/api/colors', { colors_count: count }, '色彩分析');
});

document.getElementById('btn_objects').addEventListener('click', () => {
    runVisualTask('/api/objects', {}, '物体检测');
});

document.getElementById('btn_shotscale').addEventListener('click', () => {
    runVisualTask('/api/shotscale', {}, '景别分析');
});

function renderCharts(files) {
    dom.chartBox.innerHTML = '';
    
    const map = [
        { key: 'color_palette', title: '色彩调色板' },
        { key: 'color', title: '色彩统计' },
        { key: 'objects', title: '物体检测统计' },
        { key: 'shotscale_timeline', title: '景别时间线' },
        { key: 'shotscale', title: '景别分布' },
        { key: 'subtitle', title: '字幕词云' } // 假设词云也保存在结果里
    ];

    let hasContent = false;
    map.forEach(item => {
        if (files[item.key]) {
            hasContent = true;
            const card = document.createElement('div');
            card.className = 'result-img-card';
            const src = `${files[item.key]}?t=${new Date().getTime()}`;
            card.innerHTML = `<h4>${item.title}</h4><a href="${src}" target="_blank"><img src="${src}"></a>`;
            dom.chartBox.appendChild(card);
        }
    });

    if (!hasContent) {
        dom.chartBox.innerHTML = '<div class="placeholder-text">未生成图表</div>';
    }
}