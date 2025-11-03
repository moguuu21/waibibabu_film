// SPA Navigation for sidebar + sections
(function(){
  function init(){
    const pageTitle = document.getElementById('page_title');
    const sections = {
      'video': 'section-video',
      'shotcut': 'section-shotcut',
      'colors': 'section-colors',
      'objects': 'section-objects',
      'subtitles': 'section-subtitles',
      'shotscale': 'section-shotscale',
      'faces': 'section-faces',
      'compare': 'section-compare',
      'results': 'section-results'
    };
    const titles = {
      'video': '视频处理',
      'shotcut': '镜头切分',
      'colors': '色彩分析',
      'objects': '物体检测',
      'subtitles': '字幕识别',
      'shotscale': '镜头尺度',
      'faces': '人脸识别',
      'compare': '人脸对比',
      'results': '结果预览'
    };

    function showSection(key){
      const targetId = sections[key] || sections['video'];
      document.querySelectorAll('.content-section').forEach(sec => {
        sec.classList.toggle('active', sec.id === targetId);
      });
      document.querySelectorAll('.nav-item').forEach(item => {
        const s = item.getAttribute('data-section');
        item.classList.toggle('active', s === targetId);
      });
      if (pageTitle) pageTitle.textContent = titles[key] || titles['video'];
    }

    function keyFromHash(){
      const h = (location.hash || '#video').replace('#','');
      return ['video','shotcut','colors','objects','subtitles','shotscale','faces','compare','results'].includes(h) ? h : 'video';
    }

    function applyFromHash(){ showSection(keyFromHash()); }

    // Bind nav clicks
    document.querySelectorAll('.nav-item').forEach(item => {
      item.addEventListener('click', (e) => {
        const section = item.getAttribute('data-section');
        const key = Object.keys(sections).find(k => sections[k] === section) || 'video';
        if (location.hash !== '#' + key) {
          history.pushState({}, '', '#' + key);
        }
        showSection(key);
        const sidebar = document.getElementById('sidebar');
        const backdrop = document.getElementById('sidebar_backdrop');
        if (sidebar && sidebar.classList.contains('open')) sidebar.classList.remove('open');
        if (backdrop) backdrop.classList.remove('visible');
        e.preventDefault();
      });
    });

    window.addEventListener('popstate', applyFromHash);
    window.addEventListener('hashchange', applyFromHash);
    applyFromHash();

    // Mobile menu toggle
    const mobileBtn = document.getElementById('mobile_menu_btn');
    const sidebar = document.getElementById('sidebar');
    const backdrop = document.getElementById('sidebar_backdrop');
    const toggleSidebar = () => {
      const opened = sidebar.classList.toggle('open');
      if (backdrop){ backdrop.classList.toggle('visible', opened); }
    };
    if (mobileBtn && sidebar){
      mobileBtn.addEventListener('click', toggleSidebar);
    }
    if (backdrop){
      backdrop.addEventListener('click', ()=>{
        if (sidebar.classList.contains('open')) sidebar.classList.remove('open');
        backdrop.classList.remove('visible');
      });
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else { init(); }
})();
