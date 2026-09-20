#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语义向量数据库体验脚本 (支持摄入你自己的文档)
=================================
向量数据库: ChromaDB  (本地文件持久化, 无需启动独立服务 / 无需 Docker)
向量化:     OpenAI 兼容的 Embedding API  (云端推理, 你本地不需要任何模型)

这个脚本让你"体验"四件事:
  1. 把一段文字变成一串向量 (embedding)
  2. 把很多段文字(你自己的文档)存进向量库
  3. 长文自动切分成片段, 检索更精准
  4. 用一句话去检索, 库里自动返回"语义最相近"的几条

------------------------------------------------------------------
支持的 Embedding 服务商 (都是 OpenAI 兼容接口, 一行环境变量切换, 见文件下方 EMBED_PROVIDERS):
  - 阿里云百炼  : EMBED_PROVIDER=bailian  (默认, EMBED_MODEL=text-embedding-v3)
  - 智谱 AI    : EMBED_PROVIDER=zhipu    (有免费额度, EMBED_MODEL=embedding-3)
  - OpenAI     : EMBED_PROVIDER=openai   (EMBED_MODEL=text-embedding-3-small)
  - 任意兼容端 : 直接用 EMBED_API_BASE / EMBED_MODEL 覆盖即可

------------------------------------------------------------------
对话大模型 (负责"根据检索结果生成回答", 默认 DeepSeek; 可选, 用于 --ask 模式):
  - DeepSeek : CHAT_API_BASE=https://api.deepseek.com  CHAT_MODEL=deepseek-chat
  (重要: DeepSeek 只提供对话能力, 不提供 Embedding, 所以向量化仍需上面的 EMBED_* 提供方)

------------------------------------------------------------------
运行步骤:
  1. pip install chromadb openai          (本脚本已帮你装好)
  2. 设置环境变量 (见下方 EXAMPLES)
  3. 跑内置示例:
       python vector_demo.py                # 写入内置示例并做演示查询
  4. 换成你自己的资料:
       python vector_demo.py --ingest ./docs          # 摄入一个目录(自动读 .txt/.md/.json)
       python vector_demo.py --ingest 笔记.txt        # 或单个文件
       python vector_demo.py "我想查的问题"            # 直接语义检索
       python vector_demo.py --interactive             # 交互问答
       python vector_demo.py --reset                   # 清空集合重来

EXAMPLES (Windows PowerShell, 用阿里云百炼做向量化 + DeepSeek 做回答, 国内最稳):
  $env:EMBED_API_KEY = "sk-xxxx"                                  # 向量化(百炼)
  $env:EMBED_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
  $env:EMBED_MODEL = "text-embedding-v3"
  $env:CHAT_API_KEY = "sk-deepseek-xxxx"                          # 回答(DeepSeek)
  $env:CHAT_API_BASE = "https://api.deepseek.com"
  $env:CHAT_MODEL = "deepseek-chat"
  python vector_demo.py --ingest ./docs
  python vector_demo.py --ask "Python 怎么读 Excel"

EXAMPLES (Linux / Mac):
  export EMBED_API_KEY=sk-xxxx
  export EMBED_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
  export EMBED_MODEL=text-embedding-v3
  export CHAT_API_KEY=sk-deepseek-xxxx
  export CHAT_API_BASE=https://api.deepseek.com
  export CHAT_MODEL=deepseek-chat
  python vector_demo.py --ingest ./docs
  python vector_demo.py --ask "Python 怎么读 Excel"
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from glob import glob as glob_paths

# ---- 防呆: 若没在装了 chromadb 的虚拟环境里运行, 给出明确指引 ----
try:
    from chromadb import PersistentClient
    from chromadb.utils import embedding_functions
except ImportError:
    sys.exit(
        "✗ 找不到 chromadb 模块。\n"
        "  原因: 你用的 python 解释器不是本项目虚拟环境(.venv)里的那个。\n"
        "  解决(任选其一):\n"
        "  ① 用 venv 的完整路径运行:\n"
        "     .\\.venv\\Scripts\\python.exe vector_demo.py ...\n"
        "  ② 先激活虚拟环境, 之后就能直接用 python:\n"
        "     PowerShell:  .\\.venv\\Scripts\\Activate.ps1\n"
        "     Git Bash :  source ./.venv/Scripts/activate\n"
        "     python vector_demo.py ...\n"
        "  (chromadb 已装在 ./.venv 里, 无需重新安装)"
    )

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# ========================= 读取 .env (零依赖, 不用装 python-dotenv) =========================
# Key 一律不写进源码: 项目要放 GitHub, 明文 key 一旦 push 就等于公开。
# 读取优先级: 真实环境变量 > 项目根目录的 .env 文件。
# .env 只在本地存在, 已被 .gitignore 排除, 不会进版本库。
def _load_dotenv(path):
    """极简 .env 解析: KEY=VALUE, 支持 # 注释和引号包裹。不覆盖已存在的环境变量。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:      # 环境变量优先, .env 只做兜底
                    os.environ[k] = v
    except OSError:
        pass


_load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ========================= 【Key 配置: 只从环境/.env 读】=========================
# 把 key 填到项目根目录的 .env 里(模板见 .env.example), 或设环境变量:
#   PowerShell: $env:CHAT_API_KEY = "sk-xxx"
#   Git Bash  : export CHAT_API_KEY=sk-xxx
#
# ① 百炼(阿里云) Embedding Key —— 云端向量化用; 默认走 local 本地模型, 不用填也能跑
API_KEY      = os.getenv("EMBED_API_KEY", "")
# ② DeepSeek Key          —— 负责"根据检索结果生成回答"(RAG 问答), 必须填
CHAT_API_KEY = os.getenv("CHAT_API_KEY", "")
# ============================================================================

# ============ Embedding 提供方切换 (三家都是 OpenAI 兼容接口, 改两个变量即可) ============
# 默认百炼; 想换就设 EMBED_PROVIDER, 例如:  zhipu / openai
#   zhipu : 智谱 AI(有免费额度)   $env:EMBED_PROVIDER="zhipu"  $env:EMBED_API_KEY="<你的智谱key>"
#   openai: OpenAI               $env:EMBED_PROVIDER="openai" $env:EMBED_API_KEY="sk-..."
# 拿到新家的 key 后, 只需这两行; 也可用 EMBED_API_BASE / EMBED_MODEL 精确覆盖。
EMBED_PROVIDERS = {
    "bailian": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "text-embedding-v3"),
    "zhipu":   ("https://open.bigmodel.cn/api/paas/v4",              "embedding-3"),
    "openai":  ("https://api.openai.com/v1",                        "text-embedding-3-small"),
    "local":   (None, None),   # 本地 ONNX 模型(OnnxMiniLM_L6_V2), 不经过任何云端 API
}
# 默认用 local: 本地 ONNX 模型, 不依赖任何云端 embedding 服务, 开箱即用(百炼账号当前欠费也不影响)。
# 想用云端向量化时, 设 EMBED_PROVIDER=bailian|zhipu|openai 即可切换(对应 key 另行设置)。
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "local")
_API_BASE_DEF, _MODEL_DEF = EMBED_PROVIDERS.get(EMBED_PROVIDER, EMBED_PROVIDERS["bailian"])

API_BASE     = os.getenv("EMBED_API_BASE", _API_BASE_DEF)
MODEL        = os.getenv("EMBED_MODEL", _MODEL_DEF)
CHAT_API_BASE = os.getenv("CHAT_API_BASE", "https://api.deepseek.com")
CHAT_MODEL    = os.getenv("CHAT_MODEL", "deepseek-chat")        # DeepSeek 对话模型
COLLECTION    = os.getenv("COLLECTION", "semantic_demo")
PERSIST_DIR   = os.getenv("CHROMA_DIR", "./chroma_db")

# ========================= 示例语料 (中文) =========================
SAMPLE_DOCS = [
    "Python 入门应该从哪里开始学",
    "怎样快速学会 Python 编程",
    "推荐一些 Python 初学者的学习路线",
    "Java 基础语法该怎么入门",
    "Java 和 Python 哪个更适合零基础的新手",
    "番茄炒蛋怎么做才好吃",
    "新手学做菜先从哪道菜练手比较好",
    "跑步和游泳哪个更有助于减脂",
    "每天坚持跑步对身体有什么好处",
]

DEMO_QUERIES = [
    "我想学一门编程语言",
    "怎么炒个简单的家常菜",
    "有什么运动能帮我瘦下来",
]


# ========================= 初始化 =========================
def get_embedding_function():
    """返回一个能把文本变成向量的函数。

    - EMBED_PROVIDER="local": 用 ChromaDB 内置的 ONNX 本地模型(OnnxMiniLM_L6_V2),
      首次会自动下载约 80MB 模型, 之后完全离线、不需要任何云端 Key。
    - 其他 provider(bailian/zhipu/openai): 走对应的 OpenAI 兼容云端 Embedding API。
    """
    if EMBED_PROVIDER == "local":
        print("ℹ️ 使用本地 ONNX 向量模型(ONNXMiniLM_L6_V2), 无需任何云端 Key, 首次会下载模型。")
        return embedding_functions.ONNXMiniLM_L6_V2()
    if not API_KEY or API_KEY.startswith("<"):
        sys.exit(
            "✗ 还没有填入 Embedding Key。\n"
            "  请打开 vector_demo.py, 把顶部 `API_KEY = ...` 那行替换成你的真实 Key;\n"
            "  或设置环境变量 EMBED_API_KEY(切到智谱/OpenAI 时还要对应设 EMBED_PROVIDER)。\n"
            "  不想用云端, 也可设 EMBED_PROVIDER=local 走本地模型。"
        )
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=API_KEY,
        api_base=API_BASE,
        model_name=MODEL,
    )


def get_collection():
    """连接本地持久化库, 拿到(或创建)集合。集合自带向量化函数。"""
    client = PersistentClient(path=PERSIST_DIR)
    ef = get_embedding_function()
    col = client.get_or_create_collection(
        COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},  # 用余弦距离衡量相似度
    )
    return client, col


# ========================= 文档摄入 =========================
def chunk_text(text, max_chars=500, overlap=80):
    """把长文按段落切分成若干片段, 段落过长再用滑动窗口切。"""
    text = text.strip()
    if not text:
        return []
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 1 <= max_chars:
            buf = (buf + "\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            if len(p) > max_chars:
                step = max(1, max_chars - overlap)
                for i in range(0, len(p), step):
                    chunks.append(p[i:i + max_chars])
                buf = ""
            else:
                buf = p
    if buf:
        chunks.append(buf)
    return chunks


def load_documents(path):
    """读取一个文件或目录, 返回 [(文本内容, 来源路径), ...]。
    支持 .txt / .md (纯文本) 和 .json (字符串列表 或 [{"text": ...}, ...])。"""
    p = Path(path)
    if not p.exists():
        print(f"✗ 路径不存在: {path}")
        return []
    files = []
    if p.is_dir():
        for ext in ("*.txt", "*.md", "*.json"):
            files += glob_paths(str(p / "**" / ext), recursive=True)
    else:
        files = [str(p)]
    docs = []
    for f in sorted(set(files)):
        try:
            content = Path(f).read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"  ⚠ 跳过 {f}: {e}")
            continue
        if f.endswith(".json"):
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    for item in data:
                        t = item if isinstance(item, str) else (
                            item.get("text") or item.get("content") if isinstance(item, dict) else None)
                        if t:
                            docs.append((str(t), f))
                    continue
            except Exception:
                pass  # 不是预期格式的 JSON, 当作纯文本处理
        docs.append((content, f))
    return docs


def ingest(path, max_chars=500):
    """读取文档 -> 切分 -> 写入向量库 (用 upsert, 可重复摄入不报错)。"""
    docs = load_documents(path)
    if not docs:
        print("没有读取到任何文本, 摄入中止。")
        return
    total_chars = sum(len(t) for t, _ in docs)
    print(f"读取到 {len(docs)} 个文件, 约 {total_chars} 字, 开始切分并向量化...")
    _, col = get_collection()
    ids, texts, metas = [], [], []
    for content, src in docs:
        for i, ch in enumerate(chunk_text(content, max_chars)):
            hid = hashlib.md5((src + str(i) + ch[:50]).encode("utf-8")).hexdigest()[:10]
            ids.append(f"{Path(src).stem}_{i}_{hid}")
            texts.append(ch)
            metas.append({"source": str(src), "chunk": i})
    col.upsert(ids=ids, documents=texts, metadatas=metas)
    print(f"✓ 已摄入并存入 {len(texts)} 个片段到集合 '{COLLECTION}' (保存在 {os.path.abspath(PERSIST_DIR)})。")


# ========================= 功能 =========================
def add_sample():
    _, col = get_collection()
    if col.count() > 0:
        print(f"集合 '{COLLECTION}' 里已经有 {col.count()} 条数据, 跳过写入。"
              f"用 --reset 可清空重建, 或用 --ingest 加入你自己的资料。")
        return
    ids = [f"doc_{i}" for i in range(len(SAMPLE_DOCS))]
    col.add(ids=ids, documents=SAMPLE_DOCS)
    print(f"✓ 已写入 {len(SAMPLE_DOCS)} 条示例数据到集合 '{COLLECTION}'。")


def query(q, n=3, with_source=False):
    _, col = get_collection()
    if col.count() == 0:
        print("  (库是空的, 先跑 python vector_demo.py, 或 --ingest 你的文档)")
        return
    res = col.query(query_texts=[q], n_results=min(n, col.count()))
    docs = res["documents"][0]
    dists = res["distances"][0]
    metas = res["metadatas"][0]
    print(f"\n🔍 查询: {q}")
    for i, (d, dist, m) in enumerate(zip(docs, dists, metas), 1):
        sim = 1 - dist  # 余弦距离 -> 相似度(越接近1越像)
        src = f"  [来源: {Path(m['source']).name}]" if (with_source and m and m.get("source")) else ""
        print(f"  {i}. 相似度 {sim:.3f}{src}  {d}")


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
        query(q, with_source=True)


def reset():
    client = PersistentClient(path=PERSIST_DIR)
    try:
        client.delete_collection(COLLECTION)
        print(f"✓ 已删除集合 '{COLLECTION}'。再跑一次脚本即可重新写入。")
    except Exception as e:
        print("无需删除:", e)


def ask(q, n=3):
    """RAG 问答: 先从向量库检索相关片段, 再交给对话大模型(默认 DeepSeek)生成回答。"""
    from openai import OpenAI
    if not CHAT_API_KEY or CHAT_API_KEY.startswith("<"):
        sys.exit(
            "✗ 还没有填入 DeepSeek 的 Key。\n"
            "  请打开 vector_demo.py, 把顶部 `CHAT_API_KEY = ...` 那行的 `<在此粘贴DeepSeek的KEY>`\n"
            "  替换成你的真实 DeepSeek Key; 或者设置环境变量 CHAT_API_KEY。"
        )
    _, col = get_collection()
    if col.count() == 0:
        print("  (库是空的, 先用 --ingest 摄入文档, 或先跑 python vector_demo.py 写内置示例)")
        return
    res = col.query(query_texts=[q], n_results=min(n, col.count()))
    ctx = res["documents"][0]
    metas = res["metadatas"][0]
    context = ""
    for i, (d, m) in enumerate(zip(ctx, metas), 1):
        src = m.get("source", "") if m else ""
        context += f"[片段{i} | 来源: {src}]\n{d}\n\n"
    system = ("你是一个严谨的中文助手。请只根据下面提供的资料片段回答用户问题；"
              "如果资料里没有相关信息，就明确说“资料中没有提到”。回答要简洁，"
              "并在相关句末用 [片段i] 标注引用。")
    user = f"资料:\n{context}\n问题: {q}"
    client = OpenAI(api_key=CHAT_API_KEY, base_url=CHAT_API_BASE)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.3,
    )
    print(f"\n🤖 基于向量库检索结果, 由 {CHAT_MODEL} 生成回答:\n")
    print(resp.choices[0].message.content)


# ========================= 入口 =========================
def main():
    p = argparse.ArgumentParser(description="语义向量数据库体验脚本 (ChromaDB + 云端 Embedding)")
    p.add_argument("query", nargs="?", help="直接给一个问题做语义检索")
    p.add_argument("-n", "--top", type=int, default=3, help="返回几条结果 (默认 3)")
    p.add_argument("-i", "--interactive", action="store_true", help="交互问答模式 (带来源)")
    p.add_argument("--ingest", metavar="PATH", help="摄入一个文件或目录(.txt/.md/.json)到向量库")
    p.add_argument("--chunk", type=int, default=500, help="长文切分长度(字符), 默认 500")
    p.add_argument("--ask", metavar="QUESTION", help="RAG 问答: 检索后用 DeepSeek 等对话大模型生成回答")
    p.add_argument("--reset", action="store_true", help="清空集合后退出")
    args = p.parse_args()

    if args.reset:
        reset()
        return

    if args.ask:
        ask(args.ask, args.top)
        return

    if args.ingest:
        ingest(args.ingest, args.chunk)
        if args.query:
            query(args.query, args.top, with_source=True)
        return

    if args.query:
        query(args.query, args.top, with_source=True)
        return

    if args.interactive:
        interactive()
        return

    # 默认: 写入示例数据 + 跑几个演示查询
    add_sample()
    print("\n=== 下面是几个演示查询, 注意观察返回的是不是'语义相近'的内容 ===")
    for q in DEMO_QUERIES:
        query(q, args.top)


if __name__ == "__main__":
    main()
