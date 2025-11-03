const API = {
  base: '',
  endpoint(path) { return `${this.base}${path}`; }
};

function qs(id){ return document.getElementById(id); }
function setStatus(msg){ const el = qs('status'); if (el) el.textContent = msg || ''; }
function disableAll(disabled){
  ['btn_shotcut','btn_colors','btn_objects','btn_subtitles','btn_shotscale','btn_refresh']
    .forEach(id => { const el = qs(id); if (el) el.disabled = disabled; });
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
  if (!wrap) return;
  wrap.innerHTML = '';
  if (!data || !data.data || !data.data.exists){
    wrap.innerHTML = '<div class="status">No results found. Run an analysis first.</div>';
    return;
  }
  const files = data.data.files || {};
  const mapping = [
    ['color', 'Color Analysis'],
    ['color_palette', 'Color Palette'],
    ['objects', 'Object Detection'],
    ['shotscale', 'Shot Scale Summary'],
    ['shotscale_timeline', 'Shot Scale Timeline'],
    ['subtitles_timeline', 'Subtitles Timeline']
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
    a.textContent = 'Download Subtitles (SRT)';
    a.style.display = 'inline-block';
    a.style.marginTop = '8px';
    wrap.appendChild(a);
  }
}

async function refresh(){
  const video_path = (qs('video_path') && qs('video_path').value || '').trim();
  if (!video_path){ setStatus('Please enter a local video path.'); return; }
  try{
    const data = await getJSON(API.endpoint(`/api/results?video_path=${encodeURIComponent(video_path)}`));
    resultsToCards(data);
    setStatus('Results refreshed.');
  }catch(err){ setStatus('Refresh failed: ' + err.message); }
}

async function runTask(path, payload){
  const video_path = (qs('video_path') && qs('video_path').value || '').trim();
  if (!video_path){ setStatus('Please enter a local video path.'); return; }
  disableAll(true); setStatus('Processing... This may take a while.');
  try{
    const json = await postJSON(API.endpoint(path), { video_path, ...payload });
    setStatus(json.message || 'Done.');
    resultsToCards({ data: json.results });
  }catch(err){ setStatus('Error: ' + err.message); }
  finally { disableAll(false); }
}

window.addEventListener('DOMContentLoaded', () => {
  API.base = '';
  const on = (id, fn) => { const el = qs(id); if (el) el.addEventListener('click', fn); };
  on('btn_shotcut',   ()=> runTask('/api/shotcut',   { th: parseFloat((qs('th') && qs('th').value) || '0.5') }));
  on('btn_colors',    ()=> runTask('/api/colors',    { colors_count: parseInt((qs('colors_count') && qs('colors_count').value) || '5',10) }));
  on('btn_objects',   ()=> runTask('/api/objects',   {}));
  on('btn_subtitles', ()=> runTask('/api/subtitles', { subtitle_value: parseInt((qs('subtitle_value') && qs('subtitle_value').value) || '48',10) }));
  on('btn_shotscale', ()=> runTask('/api/shotscale', {}));
  on('btn_refresh',   refresh);

  // Face features
  const bAdd = qs('btn_face_add');
  const bList = qs('btn_face_list');
  const bExt = qs('btn_face_extract');
  const bCmp = qs('btn_face_compare');
  if (bAdd) bAdd.addEventListener('click', addFaceSample);
  if (bList) bList.addEventListener('click', listKnownFaces);
  if (bExt) bExt.addEventListener('click', extractFaces);
  if (bCmp) bCmp.addEventListener('click', compareFaces);
  if (bList) listKnownFaces().catch(()=>{});
});

async function uploadFile(file){
  if (!file){ setStatus('No file selected.'); return; }
  const allowed = ['video/','application/octet-stream'];
  const type = String(file.type || '').toLowerCase();
  if (!allowed.some(p => type.startsWith(p))){ setStatus('Unsupported file type.'); return; }
  try{
    setStatus('Uploading video...');
    const fd = new FormData();
    fd.append('file', file, file.name);
    const res = await fetch(API.endpoint('/api/upload'), { method:'POST', body: fd });
    const json = await res.json().catch(()=>({ ok:false, error:'Invalid JSON' }));
    if (!res.ok || !json.ok) throw new Error(json.error || res.statusText);
    const saved = json.data && json.data.saved_path;
    if (saved){
      qs('video_path').value = saved;
      setStatus(`Uploaded: ${json.data.filename}`);
      try{ await refresh(); }catch(_){ }
    } else {
      setStatus('Upload failed: no saved path.');
    }
  }catch(err){ setStatus('Upload failed: ' + err.message); }
}

document.addEventListener('DOMContentLoaded', () => {
  const dz = document.getElementById('dropzone') || document.querySelector('.upload-area');
  const fi = document.getElementById('file_input');
  if (dz && fi){
    const stop = (e) => { e.preventDefault(); e.stopPropagation(); };
    ['dragenter','dragover'].forEach(ev => dz.addEventListener(ev, (e)=>{ stop(e); dz.classList.add('dragover'); }));
    ['dragleave','dragend','drop'].forEach(ev => dz.addEventListener(ev, (e)=>{ stop(e); dz.classList.remove('dragover'); }));
    dz.addEventListener('drop', (e)=>{
      const files = (e.dataTransfer && e.dataTransfer.files) || [];
      if (files.length > 0){ uploadFile(files[0]); }
    });
    dz.addEventListener('click', ()=> fi.click());
    fi.addEventListener('change', ()=>{ if (fi.files && fi.files[0]) uploadFile(fi.files[0]); });
  }
});

// ---------------- Face features ----------------
async function listKnownFaces(){
  const wrap = qs('faces_known'); if (!wrap) return;
  wrap.innerHTML = 'Loading...';
  try{
    const json = await getJSON(API.endpoint('/api/faces'));
    const list = json.faces || [];
    if (list.length === 0){ wrap.innerHTML = '<div class="status">No samples yet.</div>'; return; }
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
        }catch(err){ setStatus('Delete failed: ' + err.message); }
      });
    });
  }catch(err){ wrap.innerHTML = '<div class="status">Load failed: ' + err.message + '</div>'; }
}

async function addFaceSample(){
  const name = (qs('face_name') && qs('face_name').value || '').trim();
  const file = qs('face_file') && qs('face_file').files && qs('face_file').files[0];
  if (!name){ setStatus('Please enter a name.'); return; }
  if (!file){ setStatus('Please choose a sample image.'); return; }
  try{
    const fd = new FormData();
    fd.append('name', name);
    fd.append('file', file, file.name);
    const res = await fetch(API.endpoint('/api/faces/add'), { method:'POST', body: fd });
    const j = await res.json().catch(()=>({ok:false,error:'Invalid JSON'}));
    if (!res.ok || !j.ok) throw new Error(j.error || res.statusText);
    setStatus(j.message || 'Added.');
    try{ await listKnownFaces(); }catch(_){ }
  }catch(err){ setStatus('Add failed: ' + err.message); }
}

async function extractFaces(){
  const video_path = (qs('video_path') && qs('video_path').value || '').trim();
  if (!video_path){ setStatus('Please enter a local video path.'); return; }
  const wrap = qs('faces_results'); if (wrap) wrap.innerHTML = 'Processing...';
  try{
    const j = await postJSON(API.endpoint('/api/face/extract_frames'), { video_path });
    const faces = (j && j.data && j.data.faces) || [];
    if (!wrap) return;
    if (faces.length === 0){ wrap.innerHTML = '<div class="status">No faces found in keyframes.</div>'; return; }
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
          ${attrs ? `<div class=\"face-attrs\">${attrs}</div>` : ''}
        </div>
        <div class="face-add">
          <input type="text" placeholder="输入姓名后添加为样本" value="${guessed}">
          <button>Add</button>
        </div>`;
      const btn = div.querySelector('button');
      const inp = div.querySelector('input');
      btn.addEventListener('click', async ()=>{
        const name = (inp.value || '').trim();
        if (!name){ setStatus('Please enter a name first.'); return; }
        try{
          const res = await fetch(API.endpoint('/api/faces/add_by_path'), {
            method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({ image_path: f.path, name })
          });
          const jr = await res.json().catch(()=>({ok:false,error:'Invalid JSON'}));
          if (!res.ok || !jr.ok) throw new Error(jr.error || res.statusText);
          setStatus('Added to database: ' + name);
          try{ await listKnownFaces(); }catch(_){ }
        }catch(err){ setStatus('Add failed: ' + err.message); }
      });
      wrap.appendChild(div);
    }
  }catch(err){ if (wrap) wrap.innerHTML = '<div class="status">Failed: ' + err.message + '</div>'; }
}

async function compareFaces(){
  const f1 = qs('face_cmp_1') && qs('face_cmp_1').files && qs('face_cmp_1').files[0];
  const f2 = qs('face_cmp_2') && qs('face_cmp_2').files && qs('face_cmp_2').files[0];
  if (!f1 || !f2){ setStatus('Choose two images to compare.'); return; }
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
    if (info) info.textContent = `${j.message} (similarity ${j.similarity}%)`;
    setStatus('Comparison done.');
  }catch(err){ setStatus('Compare failed: ' + err.message); }
}

