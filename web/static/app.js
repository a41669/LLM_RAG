// ==========================================================================
// RAG 读书助手 前端逻辑
// 后端接口: GET /api/info | GET /api/collections | POST /api/ask
//           POST /api/upload (multipart) | POST /api/reset
// ==========================================================================

const SID = localStorage.getItem('rag_sid')
  || (localStorage.setItem('rag_sid', crypto.randomUUID()), localStorage.getItem('rag_sid'));

const chat = document.getElementById('chat');
const sources = document.getElementById('sources');
let curColl = null;   // 当前问答所用集合; null = 服务端默认书(由环境变量指定)

// ---------- 书籍下拉框 ----------
async function loadBooks() {
  try {
    const d = await fetch('/api/collections').then(r => r.json());
    const sel = document.getElementById('books');
    sel.innerHTML = '';
    const def = document.createElement('option');
    def.value = '';
    def.textContent = '📕 默认书（' + (d.default || '') + '）';
    sel.appendChild(def);
    (d.books || []).forEach(b => {
      const o = document.createElement('option');
      o.value = b.collection;
      o.textContent = '📄 ' + b.name + '（' + b.chunks + ' 段）';
      sel.appendChild(o);
    });
    if (curColl) sel.value = curColl;
  } catch (e) { /* 忽略 */ }
}

function switchBook() {
  const sel = document.getElementById('books');
  curColl = sel.value || null;
  const label = sel.options[sel.selectedIndex].textContent;
  chat.innerHTML = '';
  sources.innerHTML = '<span style="color:#8a93a0">提问后这里显示来源</span>';
  addMsg('已切换到：' + label + '。直接提问即可。', 'bot');
}

// ---------- 上传 ----------
async function uploadFile() {
  const inp = document.getElementById('file');
  if (!inp.files || !inp.files.length) return;
  const fd = new FormData();
  fd.append('file', inp.files[0]);
  addMsg('⏳ 正在上传并向量化《' + inp.files[0].name + '》…', 'bot');
  try {
    const res = await fetch('/api/upload', { method: 'POST', body: fd });
    const data = await res.json();
    if (data.error) { addMsg('上传失败: ' + data.error, 'bot'); return; }
    curColl = data.collection;
    chat.innerHTML = '';
    sources.innerHTML = '<span style="color:#8a93a0">提问后这里显示来源</span>';
    addMsg('✅ 已建库《' + data.name + '》，共 ' + data.chunks + ' 个片段，现在可以直接问它了。', 'bot');
    await loadBooks();                       // 刷新下拉框并选中新书
    document.getElementById('books').value = data.collection;
  } catch (e) { addMsg('上传失败: ' + e, 'bot'); }
  inp.value = '';
}

// ---------- 问答 ----------
function addMsg(text, cls) {
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  d.textContent = text;
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
  return d;
}

async function send() {
  const ta = document.getElementById('q'), btn = document.getElementById('send');
  const q = ta.value.trim(); if (!q) return;
  addMsg(q, 'user'); ta.value = ''; btn.disabled = true;
  const thinking = addMsg('思考中…', 'bot');
  try {
    const res = await fetch('/api/ask', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session: SID, question: q, collection: curColl })
    });
    const data = await res.json();
    chat.removeChild(thinking);
    if (data.low) addMsg('⚠️ 检索相似度偏低，资料里很可能没有相关内容，以下回答仅供参考。', 'warn');
    addMsg(data.answer, 'bot');
    renderSources(data.sources);
  } catch (e) { chat.removeChild(thinking); addMsg('请求失败: ' + e, 'bot'); }
  btn.disabled = false;
}

function renderSources(list) {
  if (!list || !list.length) { sources.innerHTML = '<span style="color:#8a93a0">未检索到片段</span>'; return; }
  sources.innerHTML = list.map(s => {
    const pct = Math.round(s.score * 100);
    const snip = s.text.length > 120 ? s.text.slice(0, 120) + '…' : s.text;
    return `<div class="src"><div>第${s.chunk}块 · 相似度 ${s.score.toFixed(4)}</div>
      <div class="bar"><i style="width:${pct}%"></i></div><div class="txt">${snip.replace(/</g, '&lt;')}</div></div>`;
  }).join('');
}

function resetChat() {
  fetch('/api/reset', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session: SID })
  }).then(() => {
    chat.innerHTML = '';
    sources.innerHTML = '<span style="color:#8a93a0">提问后这里显示来源</span>';
  });
}

document.getElementById('q').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});

loadBooks();
