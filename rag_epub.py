#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 epub 电子书做 RAG 的 Demo
===========================
  - 向量化(把文字变向量): 阿里云百炼 Embedding API  (复用 vector_demo.py 里已填好的 Key)
  - 向量库:               ChromaDB (本地文件持久化, 无需服务)
  - 回答生成(读资料作答):  DeepSeek                 (复用 vector_demo.py 里已填好的 Key)

Key 直接复用 vector_demo.py 顶部已经填好的百炼/DeepSeek Key, 你无需再填一遍。

------------------------------------------------------------------
用法 (必须用项目虚拟环境的 python 运行, 见下方):
  python rag_epub.py --ingest                # 解析电子书并向量化入库(只需第一次)
  python rag_epub.py --ask "如何高效记笔记"   # 让 DeepSeek 基于书的内容回答
  python rag_epub.py --interactive           # 连续对话模式
  python rag_epub.py --reset                  # 清空本书向量库

可覆盖的环境变量(一般不用改):
  EPUB_PATH       电子书路径        (默认 ./docs/如何学会学习.epub)
  BOOK_COLLECTION 集合名           (默认 how_to_learn_book)
  CHROMA_DIR      向量库目录        (默认 ./chroma_db)
  EMBED_MODEL     百炼 embedding 模型 (默认 text-embedding-v3)
"""

import os
import sys
import zipfile
import hashlib
import argparse
from html.parser import HTMLParser
from pathlib import Path

# ---- 防呆: 必须在装了 chromadb 的 venv 里运行 ----
try:
    import chromadb  # noqa: F401
except ImportError:
    sys.exit(
        "✗ 找不到 chromadb 模块。请使用项目虚拟环境的 python 运行:\n"
        "   .\\.venv\\Scripts\\python.exe rag_epub.py ...\n"
        "   或先激活: source ./.venv/Scripts/activate"
    )

# ============ 复用 vector_demo.py 里已填好的 Key / 配置 ============
import vector_demo  # 该模块有 `if __name__ == "__main__"` 保护, import 不会误执行其 main()
API_KEY      = vector_demo.API_KEY        # 百炼 Embedding Key
CHAT_API_KEY = vector_demo.CHAT_API_KEY   # DeepSeek Key
API_BASE     = vector_demo.API_BASE
CHAT_API_BASE = vector_demo.CHAT_API_BASE
CHAT_MODEL   = vector_demo.CHAT_MODEL
embedding_functions = vector_demo.embedding_functions
PersistentClient = vector_demo.PersistentClient
chunk_text = vector_demo.chunk_text       # 直接复用现成的切分函数

# 百炼 embedding 模型: 用正确可用的名称 (vector_demo 默认那个 qwen3.7 名字在百炼不存在)
EMBED_MODEL  = os.getenv("EMBED_MODEL", "text-embedding-v3")
COLLECTION   = os.getenv("BOOK_COLLECTION", "how_to_learn_book")
PERSIST_DIR  = os.getenv("CHROMA_DIR", "./chroma_db")
EPUB_PATH    = os.getenv("EPUB_PATH", "./docs/如何学会学习.epub")


# ========================= epub 解析 (纯标准库) =========================
class _TextExtractor(HTMLParser):
    """把 XHTML/HTML 去掉标签, 提取正文文本。"""
    _BREAK_TAGS = ("p", "br", "div", "h1", "h2", "h3", "h4", "h5", "h6",
                   "li", "tr", "section", "blockquote")

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip = True
        elif tag in self._BREAK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = False

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)

    def text(self):
        return "".join(self.parts)


def extract_epub_text(path):
    """解压 epub(本质是个 zip), 提取所有 XHTML/HTML 正文并拼接。"""
    chunks_texts = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            low = name.lower()
            if not low.endswith((".xhtml", ".html", ".htm")):
                continue
            try:
                data = z.read(name).decode("utf-8", errors="ignore")
            except Exception:
                continue
            ex = _TextExtractor()
            ex.feed(data)
            t = ex.text().strip()
            # 过滤掉太短的噪音片段(导航/目录页等)
            if len(t) > 30:
                chunks_texts.append(t)
    return "\n\n".join(chunks_texts)


# ========================= 向量库 =========================
def get_collection():
    client = PersistentClient(path=PERSIST_DIR)
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=API_KEY,
        api_base=API_BASE,
        model_name=EMBED_MODEL,
    )
    col = client.get_or_create_collection(
        COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return client, col


def ingest_epub(path=None, max_chars=600):
    path = path or EPUB_PATH
    if not Path(path).exists():
        sys.exit(f"✗ 电子书不存在: {path}")
    print(f"解析电子书: {path}")
    full = extract_epub_text(path)
    print(f"  提取正文约 {len(full)} 字")
    chunks = chunk_text(full, max_chars)
    print(f"  切分为 {len(chunks)} 个片段, 开始向量化入库 (可能需十几秒)...")
    _, col = get_collection()
    ids, texts, metas = [], [], []
    for i, ch in enumerate(chunks):
        hid = hashlib.md5(ch[:60].encode("utf-8")).hexdigest()[:10]
        ids.append(f"book_{i}_{hid}")
        texts.append(ch)
        metas.append({"source": Path(path).name, "chunk": i})
    col.upsert(ids=ids, documents=texts, metadatas=metas)
    print(f"✓ 已摄入 {len(texts)} 个片段到集合 '{COLLECTION}' "
          f"(保存在 {os.path.abspath(PERSIST_DIR)})。")


# ========================= 检索 + 问答 =========================
def ask(q, n=3):
    from openai import OpenAI
    if not CHAT_API_KEY or CHAT_API_KEY.startswith("<"):
        sys.exit("✗ 未检测到 DeepSeek Key。请先在 vector_demo.py 顶部填好 CHAT_API_KEY。")
    _, col = get_collection()
    if col.count() == 0:
        print("  (书还没入库, 请先跑: python rag_epub.py --ingest)")
        return
    res = col.query(query_texts=[q], n_results=min(n, col.count()))
    ctx = res["documents"][0]
    metas = res["metadatas"][0]
    context = ""
    for i, (d, m) in enumerate(zip(ctx, metas), 1):
        src = m.get("source", "") if m else ""
        context += f"[片段{i} | 来源: {src}]\n{d}\n\n"
    system = ("你是一个严谨的中文阅读助手。请只根据下面提供的书籍片段回答用户问题；"
              "如果资料里没有相关信息，就明确说“资料中没有提到”。回答要简洁有条理，"
              "并在相关句末用 [片段i] 标注引用。")
    user = f"书籍资料:\n{context}\n问题: {q}"
    client = OpenAI(api_key=CHAT_API_KEY, base_url=CHAT_API_BASE)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.3,
    )
    print(f"\n🤖 基于《{Path(EPUB_PATH).stem}》检索结果, 由 {CHAT_MODEL} 生成回答:\n")
    print(resp.choices[0].message.content)


def interactive():
    print("进入交互模式 (输入 quit / exit 退出):")
    while True:
        try:
            q = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            break
        ask(q)


def reset():
    client = PersistentClient(path=PERSIST_DIR)
    try:
        client.delete_collection(COLLECTION)
        print(f"✓ 已删除集合 '{COLLECTION}'。再跑一次 --ingest 即可重建。")
    except Exception as e:
        print("无需删除:", e)


def main():
    p = argparse.ArgumentParser(description="epub 电子书 RAG Demo (ChromaDB + 百炼 + DeepSeek)")
    p.add_argument("--ingest", nargs="?", const=EPUB_PATH, metavar="PATH",
                   help="解析电子书并向量化入库 (默认用 EPUB_PATH)")
    p.add_argument("--ask", metavar="QUESTION", help="基于书籍内容问答")
    p.add_argument("-n", "--top", type=int, default=3, help="检索/引用几条片段 (默认 3)")
    p.add_argument("-i", "--interactive", action="store_true", help="连续对话模式")
    p.add_argument("--reset", action="store_true", help="清空本书向量库")
    args = p.parse_args()

    if args.reset:
        reset()
        return

    if args.ingest:
        ingest_epub(args.ingest, max_chars=600)
        if args.ask:
            ask(args.ask, args.top)
        return

    if args.ask:
        ask(args.ask, args.top)
        return

    if args.interactive:
        interactive()
        return

    # 默认: 提示先入库
    print("请先摄入电子书: python rag_epub.py --ingest")
    print("然后提问:       python rag_epub.py --ask \"你的问题\"")


if __name__ == "__main__":
    main()
