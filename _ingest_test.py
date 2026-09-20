# 离线验证: 文档摄入 + 切分 + upsert + 带元数据检索 全链路
import os, shutil, hashlib
import vector_demo as v
from chromadb import PersistentClient

class DummyEF:
    def name(self): return "dummy"
    def _vec(self, t):
        dim = 8; vec = [0.0] * dim
        for c in t: vec[ord(c) % dim] += 1.0
        return vec
    def __call__(self, input): return self.embed_documents(input)
    def embed_documents(self, input): return [self._vec(t) for t in input]
    def embed_query(self, input): return [self._vec(t) for t in input]

print("=== 1) 验证文件读取 + 切分 ===")
docs = v.load_documents("./docs")
print("读取文件数:", len(docs))
for content, src in docs:
    print(f"  {os.path.basename(src)} -> {len(v.chunk_text(content, 500))} 个片段")

print("\n=== 2) 端到端(离线伪向量) ===")
if os.path.exists("./_ingest_db"): shutil.rmtree("./_ingest_db")
col = PersistentClient(path="./_ingest_db").get_or_create_collection(
    "ingest_test", embedding_function=DummyEF())
ids, texts, metas = [], [], []
for content, src in docs:
    for i, ch in enumerate(v.chunk_text(content, 500)):
        hid = hashlib.md5((src + str(i)).encode()).hexdigest()[:10]
        ids.append(f"{os.path.basename(src)}_{i}_{hid}")
        texts.append(ch); metas.append({"source": src, "chunk": i})
col.upsert(ids=ids, documents=texts, metadatas=metas)
print("upsert 片段总数:", col.count())
r = col.query(query_texts=["怎么用 Python 处理表格数据"], n_results=2)
for d, m in zip(r["documents"][0], r["metadatas"][0]):
    print(f"  命中 [{os.path.basename(m['source'])}] -> {d[:40]}")
shutil.rmtree("./_ingest_db")
print("OK: 摄入+切分+upsert+带元数据检索 全链路正常")
