#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网页版 RAG 聊天 (零额外依赖, 仅用 Python 标准库 http.server)
==========================================================
  - 复用 rag_book.py 的一切: 百炼 Embedding Key / DeepSeek Key / 向量库 / 检索逻辑
  - 多轮对话记忆: 前几轮问答作为上下文带入, DeepSeek 能"接着上文"聊
  - 相关性阈值: 检索偏弱时提示"资料可能没提到"(同 rag_book 的 --threshold)
  - 上传 txt 即时建库: 点"📁 上传书"选任意 .txt, 自动向量化并切到该集合对话(每本书独立集合)
  - 纯标准库实现, 不装 Flask/Streamlit, 避免 PyPI 下载慢

目录结构(前后端分离, 改界面不用动 Python):
  rag_web.py              本文件: HTTP 服务 + API(只放 Python)
  web/index.html          页面骨架
  web/static/style.css    样式
  web/static/app.js       前端交互

------------------------------------------------------------------
用法:
  # 默认对精简版提问(集合 learn_how_to_learn)
  ./.venv/Scripts/python.exe rag_web.py

  # 对全本提问(集合 learn_full_book)
  $env:BOOK_COLLECTION = "learn_full_book"
  ./.venv/Scripts/python.exe rag_web.py

  # 改前端时开开发模式: 每次请求重读 web/ 下的文件, 刷新浏览器即生效(不用重启服务)
  $env:RAG_WEB_DEV = "1"
  ./.venv/Scripts/python.exe rag_web.py --dev

  # 换端口 / 阈值
  $env:PORT = 8080
  ./.venv/Scripts/python.exe rag_web.py --threshold 0.5

启动后浏览器打开终端里打印的地址(默认 http://127.0.0.1:8000 )。
"""

import os
import sys
import json
import uuid
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

# 复用 rag_book 的配置与检索函数(单一真相源, Key 不用再填)
import rag_book
API_KEY       = rag_book.API_KEY
CHAT_API_KEY  = rag_book.CHAT_API_KEY
CHAT_API_BASE = rag_book.CHAT_API_BASE
CHAT_MODEL    = rag_book.CHAT_MODEL
COLLECTION    = rag_book.COLLECTION
RETRIEVE      = rag_book.retrieve
get_collection = rag_book.get_collection

PORT = int(os.getenv("PORT", "8000"))
# 用可变容器装阈值, 这样 main() 里改 --threshold 能被 Handler 读到(避免 global 声明冲突)
CONFIG = {"threshold": float(os.getenv("RELEVANCE_THRESHOLD", rag_book.RELEVANCE_THRESHOLD))}

# 每个浏览器会话的对话历史(内存里, 重启即清空)
SESSIONS = {}
MAX_HISTORY_TURNS = 6   # 最多带几轮历史进上下文

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
INDEX_FILE = os.path.join(UPLOAD_DIR, "index.json")   # 集合名 -> {name, path, chunks} 的持久化索引(重启不丢)


def load_index():
    """读取上传书籍索引 {coll: {name, path, chunks}}。"""
    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_index(idx):
    try:
        with open(INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(idx, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def list_uploads():
    """返回上传书籍列表 [{collection, name, chunks}], 按入库顺序(新在后)。"""
    idx = load_index()
    return [{"collection": c, "name": v.get("name", c), "chunks": v.get("chunks", 0)}
            for c, v in idx.items()]


# ========================= 上传解析 (Python 3.13 已移除 cgi, 手动解析 multipart) =========================
def parse_uploaded_file(raw: bytes, ctype: str):
    """从 multipart/form-data 体中取出第一个带 filename 的字段。返回 (原文件名, 文件字节)。"""
    import re
    m = re.search(r"boundary=([^;]+)", ctype or "")
    if not m:
        return None, None
    boundary = m.group(1).strip().encode("utf-8")
    delim = b"--" + boundary
    for part in raw.split(delim):
        if b"filename=" not in part:
            continue
        head, _, data = part.partition(b"\r\n\r\n")
        if not data:
            continue
        if data.endswith(b"\r\n"):
            data = data[:-2]
        head_str = head.decode("utf-8", "replace")
        fm = re.search(r'filename="([^"]*)"', head_str)
        fname = fm.group(1) if fm else "upload.txt"
        return fname, data
    return None, None


def handle_upload(raw: bytes, ctype: str):
    """保存上传的 txt 并即时向量化入库, 返回 {collection, name, chunks} 或 {error}。"""
    import re, hashlib, uuid
    fname, data = parse_uploaded_file(raw, ctype)
    if not data:
        return {"error": "未收到文件"}
    # 仅接受文本类文件
    if not (fname.lower().endswith(".txt") or ctype.lower().startswith("text/")):
        return {"error": "仅支持 .txt 文本文件"}
    # 用 文件名+大小 生成稳定集合名(中文名也不影响 chroma 集合命名)
    coll = "upload_" + hashlib.md5((fname + str(len(data))).encode("utf-8")).hexdigest()[:12]
    # 落盘(文件名做安全化, 仅保留基本字符)
    safe = re.sub(r"[^A-Za-z0-9_.一-鿿\-]", "_", fname) or "upload.txt"
    save_path = os.path.join(UPLOAD_DIR, safe)
    try:
        with open(save_path, "wb") as f:
            f.write(data)
    except Exception as e:
        return {"error": f"保存文件失败: {e}"}
    # 即时入库(embedding 走 vector_demo 当前提供方, 默认 local 离线)
    try:
        rag_book.ingest_book(save_path, max_chars=int(os.getenv("CHUNK_CHARS", "600")),
                             collection=coll)
    except Exception as e:
        return {"error": f"向量化入库失败: {type(e).__name__}: {str(e)[:200]}"}
    # 持久化索引: 集合名 <-> 书名(重启后仍可切换)
    chunks = rag_book.get_collection(coll)[1].count()
    idx = load_index()
    idx[coll] = {"name": fname, "path": save_path, "chunks": chunks}
    save_index(idx)
    return {"collection": coll, "name": fname, "chunks": chunks}



# ========================= 核心: 带记忆的问答 =========================
def answer_with_memory(session_id, question, n=3, threshold=CONFIG["threshold"], collection=None):
    """检索 + 多轮记忆 + DeepSeek 生成。返回 (answer, sources, low_relevance)。

    collection: 指定从哪个集合检索; 不传用全局默认 COLLECTION(支持上传新书后切集合)。
    """
    from openai import OpenAI
    if not CHAT_API_KEY or CHAT_API_KEY.startswith("<"):
        return "✗ 未检测到 DeepSeek Key。请先在 vector_demo.py 顶部填好 CHAT_API_KEY。", [], False

    sess = SESSIONS.setdefault(session_id, {"history": []})

    r = RETRIEVE(question, n, collection=collection)
    if r is None:
        return "（这本书还没入库，请先上传或跑：python rag_book.py --ingest）", [], False

    scores = r["scores"]
    top = max(scores)
    low = top < threshold
    sources = [
        {"score": s, "chunk": m.get("chunk", "?"), "text": d}
        for d, m, s in zip(r["docs"], r["metas"], scores)
    ]

    # 系统提示: 严谨、只依据资料、标注引用; 弱相关时更谨慎
    system = (
        "你是一个严谨的中文阅读助手。请只根据下面提供的书籍片段回答用户问题；"
        "如果片段里没有相关信息，就明确说“资料中没有提到”。回答要简洁有条理，"
        "并在相关句末用 [片段i] 标注引用。"
    )
    if low:
        system += ("\n注意：本次检索到的片段相似度都偏低，资料很可能没有相关信息，"
                   "请直接说“资料中没有提到”，不要编造。")

    # 组装消息: 系统 + 历史多轮 + 本轮(带检索到的资料)
    messages = [{"role": "system", "content": system}]
    for turn in sess["history"][-MAX_HISTORY_TURNS:]:
        messages.append({"role": "user", "content": turn["q"]})
        messages.append({"role": "assistant", "content": turn["a"]})
    user_msg = f"书籍资料:\n{r['context']}\n\n（以上为第1~{n}个相关片段，请优先依据它们作答）\n问题: {question}"
    messages.append({"role": "user", "content": user_msg})

    client = OpenAI(api_key=CHAT_API_KEY, base_url=CHAT_API_BASE)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.3,
    )
    answer = resp.choices[0].message.content

    # 记入历史(供下一轮"接着聊")
    sess["history"].append({"q": question, "a": answer})
    return answer, sources, low


# ========================= 前端静态文件 (web/ 目录) =========================
# 页面/CSS/JS 全部放在 web/ 下, 改界面不用再动 Python:
#     web/index.html          页面骨架
#     web/static/style.css    样式
#     web/static/app.js       交互逻辑
# 开发模式(环境变量 RAG_WEB_DEV=1 或加 --dev): 每次请求重读文件, 改完刷新浏览器即可生效;
# 生产模式(默认): 读一次缓存进内存, 少做磁盘 IO。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR  = os.path.join(BASE_DIR, "web")
DEV      = os.getenv("RAG_WEB_DEV", "0").lower() not in ("0", "", "false", "no")

MIME = {
    ".html": "text/html; charset=utf-8",
    ".css":  "text/css; charset=utf-8",
    ".js":   "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg":  "image/svg+xml",
    ".png":  "image/png",
    ".ico":  "image/x-icon",
}
_ASSET_CACHE = {}


def read_asset(rel_path: str):
    """安全读取 web/ 下的静态文件, 返回 (bytes, mime); 找不到返回 (None, None)。

    rel_path 为空 -> 返回 index.html。带目录穿越防护: 只允许访问 WEB_DIR 内的文件。
    """
    rel = (rel_path or "").lstrip("/") or "index.html"
    full = os.path.abspath(os.path.join(WEB_DIR, rel))
    if os.path.commonpath([full, WEB_DIR]) != WEB_DIR:   # 挡住 ../../etc/passwd 之类
        return None, None
    mime = MIME.get(os.path.splitext(full)[1].lower(), "application/octet-stream")
    if not DEV and rel in _ASSET_CACHE:
        return _ASSET_CACHE[rel], mime
    try:
        with open(full, "rb") as f:
            data = f.read()
    except OSError:
        return None, None
    if not DEV:
        _ASSET_CACHE[rel] = data
    return data, mime


# ========================= HTTP 服务 =========================
class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload, ctype="application/json; charset=utf-8"):
        data = payload if isinstance(payload, (bytes, bytearray)) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        # 非 /api 的请求一律当静态文件处理: 页面 / css / js 都在 web/ 下
        if not path.startswith("/api/"):
            data, mime = read_asset("" if path in ("", "/") else path)
            if data is None:
                self._send(404, {"error": "not found"})
            else:
                self._send(200, data, mime)
            return

        if path == "/api/info":
            self._send(200, {"collection": COLLECTION, "model": CHAT_MODEL})
        elif path == "/api/collections":
            self._send(200, {"default": COLLECTION, "books": list_uploads()})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        ctype = self.headers.get("Content-Type", "")
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)

        # 文件上传走 multipart, 不按 JSON 解析
        if "multipart/form-data" in ctype:
            try:
                res = handle_upload(raw, ctype)
                code = 200 if "error" not in res else 400
                self._send(code, res)
            except Exception as e:
                self._send(500, {"error": f"上传处理异常: {type(e).__name__}: {str(e)[:200]}"})
            return

        try:
            body = json.loads(raw or b"{}")
        except Exception:
            self._send(400, {"error": "请求体不是合法 JSON"})
            return
        if path == "/api/info":
            self._send(200, {"collection": COLLECTION})
        elif path == "/api/ask":
            try:
                answer, srcs, low = answer_with_memory(
                    body.get("session", "default"),
                    body.get("question", ""),
                    n=int(body.get("top", 3)),
                    collection=body.get("collection"),   # 上传新书后按集合检索
                )
                self._send(200, {"answer": answer, "sources": srcs, "low": low})
            except Exception as e:
                # 检索/生成链路出错(常见于 embedding 服务欠费或网络问题), 返回可读错误而非 500 白屏
                self._send(200, {"answer": f"⚠️ 调用出错：{type(e).__name__}: {str(e)[:300]}",
                                 "sources": [], "low": True})
        elif path == "/api/reset":
            SESSIONS.pop(body.get("session", "default"), None)
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": "not found"})

    def log_message(self, *a):
        pass  # 安静一点


def main():
    p = argparse.ArgumentParser(description="网页版 RAG 读书助手 (标准库 http.server)")
    p.add_argument("--threshold", type=float, default=CONFIG["threshold"], help="相关性阈值(默认 0.5)")
    p.add_argument("--port", type=int, default=PORT, help="端口(默认 8000)")
    p.add_argument("--dev", action="store_true",
                   help="开发模式: 每次请求重读 web/ 下的文件, 改前端刷新即生效(不用重启)")
    args = p.parse_args()
    CONFIG["threshold"] = args.threshold
    port = args.port
    if args.dev:
        globals()["DEV"] = True      # 关掉静态文件缓存

    if not CHAT_API_KEY or CHAT_API_KEY.startswith("<"):
        sys.exit("✗ 未检测到 DeepSeek Key。请先在 vector_demo.py 顶部填好 CHAT_API_KEY。")

    url = f"http://127.0.0.1:{port}"
    print(f"📚 RAG 读书助手已启动: {url}")
    print(f"   当前默认集合: {COLLECTION}  (浏览器里点'上传书'可即时建库任意 txt)")
    print(f"   前端目录: {WEB_DIR}" + ("  [开发模式: 改完刷新即可]" if DEV else "  [缓存模式: 改前端需重启, 或加 --dev]"))
    print(f"   在浏览器打开上面的地址即可对话。Ctrl+C 退出。")
    try:
        ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\n已退出。")


if __name__ == "__main__":
    main()
