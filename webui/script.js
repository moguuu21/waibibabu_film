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
  // Face UI bindings
  const bAdd = qs('btn_face_add');
  const bList = qs('btn_face_list');
  const bExt = qs('btn_face_extract');
  const bCmp = qs('btn_face_compare');
  if (bAdd) bAdd.addEventListener('click', addFaceSample);
  if (bList) bList.addEventListener('click', listKnownFaces);
  if (bExt) bExt.addEventListener('click', extractFaces);
  if (bCmp) bCmp.addEventListener('click', compareFaces);
  // Initial faces list
  if (bList) listKnownFaces().catch(()=>{});
});

async function uploadFile(file){
  if (!file){ setStatus('未选择文件'); return; }
  const allowed = ['video/','application/octet-stream'];
  const type = String(file.type || '').toLowerCase();
  if (!allowed.some(p => type.startsWith(p))){
    setStatus('不支持的文件类型');
    return;
  }
  try{
    setStatus('正在上传视频...');
    const fd = new FormData();
    fd.append('file', file, file.name);
    const res = await fetch(API.endpoint('/api/upload'), { method:'POST', body: fd });
    const json = await res.json().catch(()=>({ ok:false, error:'Invalid JSON' }));
    if (!res.ok || !json.ok) throw new Error(json.error || res.statusText);
    const saved = json.data?.saved_path;
    if (saved){
      qs('video_path').value = saved;
      setStatus(`上传完成：${json.data.filename}`);
      try{ await refresh(); }catch(_){ }
    } else {
      setStatus('上传失败：未返回保存路径');
    }
  }catch(err){
    setStatus('上传失败：' + err.message);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const dz = qs('dropzone');
  const fi = qs('file_input');
  if (dz && fi){
    const stop = (e) => { e.preventDefault(); e.stopPropagation(); };
    ['dragenter','dragover'].forEach(ev => dz.addEventListener(ev, (e)=>{ stop(e); dz.classList.add('dragover'); }));
    ['dragleave','drop'].forEach(ev => dz.addEventListener(ev, (e)=>{ stop(e); dz.classList.remove('dragover'); }));
    dz.addEventListener('drop', (e)=>{
      const files = e.dataTransfer?.files || [];
      if (files.length > 0){ uploadFile(files[0]); }
    });
    dz.addEventListener('click', ()=> fi.click());
    fi.addEventListener('change', ()=>{ if (fi.files && fi.files[0]) uploadFile(fi.files[0]); });
  }
});

// ---------------- Face features ----------------
async function listKnownFaces(){
  const wrap = qs('faces_known'); if (!wrap) return;
  wrap.innerHTML = '载入中...';
  try{
    const json = await getJSON(API.endpoint('/api/faces'));
    const list = json.faces || [];
    if (list.length === 0){ wrap.innerHTML = '<div class="status">暂无样本</div>'; return; }
    wrap.innerHTML = '';
    for (const it of list){
      const div = document.createElement('div');
      div.className = 'face-item';
      div.innerHTML = `
        <img loading="lazy" src="${it.url}" alt="${it.name}">
        <div class="face-meta">
          <div class="face-name">${it.name}</div>
          <button class="danger" data-file="${it.filename}">删除</button>
        </div>`;
      wrap.appendChild(div);
    }
    wrap.querySelectorAll('button.danger').forEach(btn => {
      btn.addEventListener('click', async () => {
        const file = btn.getAttribute('data-file');
        if (!file) return;
        try{
          const res = await fetch(API.endpoint('/api/faces/' + encodeURIComponent(file)), { method:'DELETE' });
          const j = await res.json().catch(()=>({ok:false,error:'Invalid JSON'}));
          if (!res.ok || !j.ok) throw new Error(j.error || res.statusText);
          await listKnownFaces();
        }catch(err){ setStatus('删除失败：' + err.message); }
      });
    });
  }catch(err){ wrap.innerHTML = '<div class="status">加载失败：' + err.message + '</div>'; }
}

async function addFaceSample(){
  const name = (qs('face_name')?.value || '').trim();
  const file = qs('face_file')?.files?.[0];
  if (!name){ setStatus('请填写姓名'); return; }
  if (!file){ setStatus('请选择样本图片'); return; }
  try{
    const fd = new FormData();
    fd.append('name', name);
    fd.append('file', file, file.name);
    const res = await fetch(API.endpoint('/api/faces/add'), { method:'POST', body: fd });
    const j = await res.json().catch(()=>({ok:false,error:'Invalid JSON'}));
    if (!res.ok || !j.ok) throw new Error(j.error || res.statusText);
    setStatus(j.message || '已添加');
    try{ await listKnownFaces(); }catch(_){ }
  }catch(err){ setStatus('添加失败：' + err.message); }
}

async function extractFaces(){
  const video_path = qs('video_path').value.trim();
  if (!video_path){ setStatus('请填写本地视频路径'); return; }
  const wrap = qs('faces_results'); if (wrap) wrap.innerHTML = '处理中...';
  try{
    const j = await postJSON(API.endpoint('/api/face/extract_frames'), { video_path });
    const faces = j?.data?.faces || [];
    if (!wrap) return;
    if (faces.length === 0){ wrap.innerHTML = '<div class="status">未在关键帧中检测到人脸</div>'; return; }
    wrap.innerHTML = '';
    for (const f of faces){
      const guessed = (f.name && f.name !== '未知人物') ? f.name : '';
      const div = document.createElement('div');
      div.className = 'face-item';
      const attrs = (f.attributes || []).map(a => Array.isArray(a)? `${a[0]}(${a[1]})` : a).join(' · ');
      div.innerHTML = `
        <img loading="lazy" src="${f.url}" alt="face">
        <div class="face-meta">
          <div class="face-name">${f.name || '未知人物'}${f.confidence ? ` (${f.confidence}%)` : ''}</div>
          ${attrs ? `<div class="face-attrs">${attrs}</div>` : ''}
        </div>
        <div class="face-add">
          <input type="text" placeholder="输入姓名后添加为样本" value="${guessed}">
          <button>添加为样本</button>
        </div>`;
      const btn = div.querySelector('button');
      const inp = div.querySelector('input');
      btn.addEventListener('click', async ()=>{
        const name = (inp.value || '').trim();
        if (!name){ setStatus('请输入姓名后再添加'); return; }
        try{
          const res = await fetch(API.endpoint('/api/faces/add_by_path'), {
            method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({ image_path: f.path, name })
          });
          const jr = await res.json().catch(()=>({ok:false,error:'Invalid JSON'}));
          if (!res.ok || !jr.ok) throw new Error(jr.error || res.statusText);
          setStatus('已添加到样本库：' + name);
          try{ await listKnownFaces(); }catch(_){ }
        }catch(err){ setStatus('添加失败：' + err.message); }
      });
      wrap.appendChild(div);
    }
  }catch(err){ if (wrap) wrap.innerHTML = '<div class="status">失败：' + err.message + '</div>'; }
}

async function compareFaces(){
  const f1 = qs('face_cmp_1')?.files?.[0];
  const f2 = qs('face_cmp_2')?.files?.[0];
  if (!f1 || !f2){ setStatus('请选择两张图片用于对比'); return; }
  try{
    const fd = new FormData();
    fd.append('file1', f1, f1.name);
    fd.append('file2', f2, f2.name);
    const res = await fetch(API.endpoint('/api/face/compare'), { method:'POST', body: fd });
    const j = await res.json().catch(()=>({ok:false,error:'Invalid JSON'}));
    if (!res.ok || !j.ok) throw new Error(j.error || res.statusText);
    const img = qs('face_compare_preview');
    const info = qs('face_compare_text');
    if (img && j.image_url) img.src = j.image_url;
    if (info) info.textContent = `${j.message}（相似度 ${j.similarity}%）`;
    setStatus('对比完成');
  }catch(err){ setStatus('对比失败：' + err.message); }
}
