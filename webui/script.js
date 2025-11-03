const API = {
  base: '',
  endpoint(path) { return `${this.base}${path}`; }
};

function qs(id){ return document.getElementById(id); }
function setStatus(msg){ qs('status').textContent = msg || ''; }
function disableAll(disabled){
  for (const id of ['btn_shotcut','btn_colors','btn_objects','btn_subtitles','btn_shotscale','btn_refresh']) {
    const el = qs(id); if (el) el.disabled = disabled;
  }
}

async function postJSON(url, payload){
  const res = await fetch(url, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
  const json = await res.json().catch(()=>({ ok:false, error:'Invalid JSON' }));
  if (!res.ok || !json.ok) throw new Error(json.error || res.statusText);
  return json;
}

async function getJSON(url){
  const res = await fetch(url);
  const json = await res.json().catch(()=>({ ok:false, error:'Invalid JSON' }));
  if (!res.ok || !json.ok) throw new Error(json.error || res.statusText);
  return json;
}

function resultsToCards(data){
  const wrap = qs('results');
  wrap.innerHTML = '';
  if (!data?.data?.exists){
    wrap.innerHTML = '<div class="status">未找到结果目录，请先运行任一分析。</div>';
    return;
  }
  const files = data.data.files || {};
  const mapping = [
    ['color', '色彩分析'],
    ['color_palette', '色彩调色板'],
    ['objects', '物体检测'],
    ['shotscale', '镜头尺度统计'],
    ['shotscale_timeline', '镜头尺度时间线'],
    ['subtitles_timeline', '字幕时间线']
  ];
  for (const [key, title] of mapping){
    if (files[key]){
      const div = document.createElement('div');
      div.className = 'result-item';
      div.innerHTML = `<h3>${title}</h3><img loading="lazy" src="${files[key]}" alt="${title}" />`;
      wrap.appendChild(div);
    }
  }
  if (files['subtitle_srt']){
    const a = document.createElement('a');
    a.href = files['subtitle_srt'];
    a.textContent = '下载字幕（SRT）';
    a.style.display = 'inline-block';
    a.style.marginTop = '8px';
    wrap.appendChild(a);
  }
}

async function refresh(){
  const video_path = qs('video_path').value.trim();
  if (!video_path){ setStatus('请填写本地视频路径'); return; }
  try{
    const data = await getJSON(API.endpoint(`/api/results?video_path=${encodeURIComponent(video_path)}`));
    resultsToCards(data);
    setStatus('结果已刷新');
  }catch(err){ setStatus('刷新失败：' + err.message); }
}

async function runTask(path, payload){
  const video_path = qs('video_path').value.trim();
  if (!video_path){ setStatus('请填写本地视频路径'); return; }
  disableAll(true); setStatus('处理中... 这可能需要较长时间。');
  try{
    const json = await postJSON(API.endpoint(path), { video_path, ...payload });
    setStatus(json.message || '完成');
    resultsToCards({ data: json.results });
  }catch(err){ setStatus('出错：' + err.message); }
  finally { disableAll(false); }
}

window.addEventListener('DOMContentLoaded', () => {
  // 默认同源（服务器同时提供前端和API）
  API.base = '';
  qs('btn_shotcut').addEventListener('click', ()=> runTask('/api/shotcut', { th: parseFloat(qs('th').value || '0.5') }));
  qs('btn_colors').addEventListener('click', ()=> runTask('/api/colors', { colors_count: parseInt(qs('colors_count').value || '5',10) }));
  qs('btn_objects').addEventListener('click', ()=> runTask('/api/objects', {}));
  qs('btn_subtitles').addEventListener('click', ()=> runTask('/api/subtitles', { subtitle_value: parseInt(qs('subtitle_value').value || '48',10) }));
  qs('btn_shotscale').addEventListener('click', ()=> runTask('/api/shotscale', {}));
  qs('btn_refresh').addEventListener('click', refresh);
});

