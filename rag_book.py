#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从一本 TXT 书做 RAG 的最小 Demo
================================
  - 向量化(把文字变向量): 阿里云百炼 Embedding API  (直接复用 vector_demo.py 里填好的 Key)
  - 向量库:               ChromaDB (本地文件持久化, 无需服务/Docker)
  - 回答生成(读资料作答): DeepSeek                 (直接复用 vector_demo.py 里填好的 Key)

Key 直接 import vector_demo 复用，你无需再填一遍。

------------------------------------------------------------------
用法 (必须用项目虚拟环境的 python 跑):
  ./.venv/Scripts/python.exe rag_book.py --ingest                 # 把书切分+向量化入库(只需第一次)
  ./.venv/Scripts/python.exe rag_book.py --ask "如何高效记笔记"    # 让 DeepSeek 基于书的内容回答
  ./.venv/Scripts/python.exe rag_book.py --interactive            # 连续对话模式
  ./.venv/Scripts/python.exe rag_book.py --reset                  # 清空本书向量库

可覆盖的环境变量(一般不用改):
  BOOK_PATH      书文件路径          (默认 ./docs/如何学会学习.txt)
  BOOK_COLLECTION 集合名            (默认 learn_how_to_learn)
  CHROMA_DIR     向量库目录          (默认 ./chroma_db)
  EMBED_MODEL    百炼 embedding 模型 (默认 text-embedding-v3)
"""

import os
import sys
import hashlib
import argparse
from pathlib import Path

# ---- 防呆: 必须在装了 chromadb 的 venv 里运行 ----
try:
    import chromadb  # noqa: F401
except ImportError:
    sys.exit(
        "✗ 找不到 chromadb 模块。请使用项目虚拟环境的 python 运行:\n"
        "   .\\.venv\\Scripts\\python.exe rag_book.py ...\n"
        "   或先激活: source ./.venv/Scripts/activate"
    )

# ============ 复用 vector_demo.py 里已填好的 Key / 配置 ============
import vector_demo  # 该模块有 `if __name__ == "__main__"` 保护, import 不会误执行其 main()
API_KEY       = vector_demo.API_KEY        # 百炼 Embedding Key
CHAT_API_KEY  = vector_demo.CHAT_API_KEY   # DeepSeek Key
API_BASE      = vector_demo.API_BASE
CHAT_API_BASE = vector_demo.CHAT_API_BASE
CHAT_MODEL    = vector_demo.CHAT_MODEL
embedding_functions = vector_demo.embedding_functions
PersistentClient    = vector_demo.PersistentClient
chunk_text = vector_demo.chunk_text        # 直接复用现成的切分函数
MODEL       = vector_demo.MODEL            # embedding 模型名(随 EMBED_PROVIDER 切换, 如 zhipu->embedding-3)

COLLECTION   = os.getenv("BOOK_COLLECTION", "learn_how_to_learn")
PERSIST_DIR  = os.getenv("CHROMA_DIR", "./chroma_db")
BOOK_PATH    = os.getenv("BOOK_PATH", "./docs/如何学会学习.txt")
# 相关性阈值: 最高相似度低于该值, 视为"资料很可能没提到"。余弦相似度, 范围 0~1。
# 不同 embedding 模型的相似度量级不同, 故按提供方给不同默认值(可用 --threshold 或 RELEVANCE_THRESHOLD 覆盖):
#   - bailian/zhipu/openai(云端大模型向量): 本书内相关 ~0.66+, 离题 ~0.45- → 取 0.5
#   - local(本地 MiniLM ONNX): 量级偏低且更窄, 本书内相关 ~0.38-0.43, 离题 ~0.32-0.49 → 取 0.35
# 说明: 该阈值只是"提示", DeepSeek 仍会独立判断上下文并如实说"资料中没有提到", 所以阈值偏宽松不会漏答, 只是少告警。
_PROVIDER_DEFAULT_THRESHOLD = {"local": "0.35"}
RELEVANCE_THRESHOLD = float(os.getenv(
    "RELEVANCE_THRESHOLD",
    _PROVIDER_DEFAULT_THRESHOLD.get(vector_demo.EMBED_PROVIDER, "0.5"),
))


# ========================= 读 TXT 书 =========================
def read_book(path):
    """读取一本 txt 书, 自动尝试 utf-8 / gbk 编码, 返回纯文本。"""
    p = Path(path)
    if not p.exists():
        sys.exit(f"✗ 书文件不存在: {path}")
    raw = p.read_bytes()
    text = None
    for enc in ("utf-8", "gbk", "gb18030"):
        try:
            text = raw.decode(enc)
            break
        except Exception:
            continue
    if text is None:
        text = raw.decode("utf-8", errors="ignore")
    # 简单清洗: 把 \r\n / \r 统一成 \n, 去掉行尾空格, 合并多余空行
    lines = [ln.rstrip() for ln in text.split("\n")]
    out = []
    blank = 0
    for ln in lines:
        if ln.strip() == "":
            blank += 1
            if blank <= 1:
                out.append("")
        else:
            blank = 0
            out.append(ln)
    return "\n".join(out).strip()


# ========================= 向量库 =========================
def get_collection(collection=None):
    """连接本地持久化库, 拿到(或创建)指定集合。集合自带向量化函数。

    collection: 集合名; 不传则用全局 COLLECTION(支持上传新书建独立集合)。
    """
    collection = collection or COLLECTION
    client = PersistentClient(path=PERSIST_DIR)
    ef = vector_demo.get_embedding_function()   # 随 EMBED_PROVIDER 切换(local/cloud)
    col = client.get_or_create_collection(
        collection,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return client, col


def ingest_book(path=None, max_chars=500, batch=10, collection=None):
    """读取书 -> 切分 -> 分批写入向量库 (upsert, 可重复跑)。

    batch: 每批提交几条。百炼 Embedding 接口单次批量上限为 10 条,
           所以这里默认按 10 条一批提交, 避免触发 400 报错。
    collection: 写入哪个集合; 不传用全局 COLLECTION。
    """
    path = path or BOOK_PATH
    collection = collection or COLLECTION
    print(f"读取书籍: {path}  -> 集合: {collection}")
    full = read_book(path)
    print(f"  提取正文约 {len(full)} 字")
    chunks = chunk_text(full, max_chars)
    print(f"  切分为 {len(chunks)} 个片段, 开始分批向量化入库(每批 {batch} 条)...")
    _, col = get_collection(collection)
    total = 0
    for start in range(0, len(chunks), batch):
        batch_chunks = chunks[start:start + batch]
        ids, texts, metas = [], [], []
        for i, ch in enumerate(batch_chunks):
            idx = start + i
            hid = hashlib.md5(ch[:60].encode("utf-8")).hexdigest()[:10]
            ids.append(f"book_{idx}_{hid}")
            texts.append(ch)
            metas.append({"source": Path(path).name, "chunk": idx})
        col.upsert(ids=ids, documents=texts, metadatas=metas)
        total += len(texts)
        print(f"    ✓ 已写入 {total}/{len(chunks)} 个片段")
    print(f"✓ 全部完成: 已摄入 {total} 个片段到集合 '{collection}' "
          f"(保存在 {os.path.abspath(PERSIST_DIR)})。")


# ========================= 检索 + 问答 =========================
def retrieve(q, n=3, collection=None):
    """从向量库检索与问题最相关的 n 个片段。

    返回 dict:
      context : 拼好的"片段1...片段n"上下文字符串(给大模型用)
      docs    : 片段文本列表
      metas   : 每个片段的元数据
      scores  : 每个片段的相似度 (1 - 余弦距离, 越接近 1 越相关)
    """
    _, col = get_collection(collection)
    if col.count() == 0:
        return None
    res = col.query(query_texts=[q], n_results=min(n, col.count()))
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    scores = [round(1 - d, 4) for d in dists]
    context = ""
    for i, (d, m) in enumerate(zip(docs, metas), 1):
        src = m.get("source", "") if m else ""
        context += f"[片段{i} | 来源: {src} | 第{m.get('chunk', '?')}块 | 相似度 {scores[i-1]}]\n{d}\n\n"
    return {"context": context, "docs": docs, "metas": metas, "scores": scores}


def show_retrieval(q, n=3, threshold=RELEVANCE_THRESHOLD):
    """只做检索并展示(top-N 片段 + 相似度), 不调用大模型。用于看清 RAG 的'检索'这一步。

    若最高相似度低于 threshold, 额外打印一句"资料很可能没提到"的提醒。
    """
    r = retrieve(q, n)
    if r is None:
        print("  (书还没入库, 请先跑: python rag_book.py --ingest)")
        return
    print(f"\n🔍 检索: {q}")
    for i, (d, m, s) in enumerate(zip(r["docs"], r["metas"], r["scores"]), 1):
        bar = "█" * int(s * 20)
        print(f"  {i}. 相似度 {s:.4f} {bar}  [第{m.get('chunk','?')}块]")
        print(f"     {d[:120]}{'…' if len(d) > 120 else ''}")
    top = max(r["scores"])
    if top < threshold:
        print(f"\n⚠️ 最高相似度仅 {top:.4f}, 低于阈值 {threshold}。"
              f"资料里很可能没有相关内容, 建议换种问法, 或确认这本书是否覆盖了该话题。")


def ask(q, n=3, threshold=RELEVANCE_THRESHOLD):
    """RAG 问答: 先检索相关片段, 再交给 DeepSeek 生成回答。"""
    from openai import OpenAI
    if not CHAT_API_KEY or CHAT_API_KEY.startswith("<"):
        sys.exit("✗ 未检测到 DeepSeek Key。请先在 vector_demo.py 顶部填好 CHAT_API_KEY。")
    r = retrieve(q, n)
    if r is None:
        print("  (书还没入库, 请先跑: python rag_book.py --ingest)")
        return
    # 先把"检索到了什么"透出来, 让你看清 RAG 第一步
    show_retrieval(q, n, threshold)
    # 若检索整体偏弱, 把"资料可能没提到"这件事也写进 system, 让模型更谨慎
    extra = ""
    if max(r["scores"]) < threshold:
        extra = "\n注意: 本次检索到的片段相似度都偏低, 资料很可能没有相关信息, 请直接说“资料中没有提到”。"
    system = (
        "你是一个严谨的中文阅读助手。请只根据下面提供的书籍片段回答用户问题；"
        "如果片段里没有相关信息，就明确说“资料中没有提到”。回答要简洁有条理，"
        "并在相关句末用 [片段i] 标注引用。"
        + extra
    )
    user = f"书籍资料:\n{r['context']}\n问题: {q}"
    client = OpenAI(api_key=CHAT_API_KEY, base_url=CHAT_API_BASE)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.3,
    )
    print(f"\n🤖 基于《{Path(BOOK_PATH).stem}》检索结果, 由 {CHAT_MODEL} 生成回答:\n")
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
    p = argparse.ArgumentParser(description="TXT 书籍 RAG Demo (ChromaDB + 百炼 + DeepSeek)")
    p.add_argument("--ingest", nargs="?", const=BOOK_PATH, metavar="PATH",
                   help="读取书并向量化入库 (默认用 BOOK_PATH)")
    p.add_argument("--ask", metavar="QUESTION", help="基于书籍内容问答(含检索+DeepSeek生成)")
    p.add_argument("--search", metavar="QUESTION", help="只做语义检索并展示 top-N 片段+相似度(不调大模型)")
    p.add_argument("-n", "--top", type=int, default=3, help="检索/引用几条片段 (默认 3)")
    p.add_argument("-i", "--interactive", action="store_true", help="连续对话模式")
    p.add_argument("--reset", action="store_true", help="清空本书向量库")
    p.add_argument("--chunk", type=int, default=int(os.getenv("CHUNK_CHARS", "500")),
                   help="入库切分长度(字符), 默认 500; 换全本可调大如 800")
    p.add_argument("--threshold", type=float, default=RELEVANCE_THRESHOLD,
                   help="相关性阈值(余弦相似度, 默认 0.35); 最高相似度低于它则提示资料可能没提到")
    args = p.parse_args()

    if args.reset:
        reset()
        if not (args.ingest or args.ask or args.search):
            return

    if args.ingest:
        ingest_book(args.ingest, max_chars=args.chunk)
        if args.ask:
            ask(args.ask, args.top, args.threshold)
        elif args.search:
            show_retrieval(args.search, args.top, args.threshold)
        return

    if args.search:
        show_retrieval(args.search, args.top, args.threshold)
        return

    if args.ask:
        ask(args.ask, args.top, args.threshold)
        return

    if args.interactive:
        interactive()
        return

    # 默认: 提示先入库
    print("请先摄入书: python rag_book.py --ingest")
    print("然后提问:   python rag_book.py --ask \"你的问题\"")
    print("只看检索:   python rag_book.py --search \"关键词\"")


if __name__ == "__main__":
    main()
