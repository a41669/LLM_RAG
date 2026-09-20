#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EPUB 解析器 (纯 Python 标准库实现, 无需第三方依赖)
====================================================
Python 没有内置 "读 epub" 的模块, 但完全支持解析:
EPUB 本质上是一个 ZIP 压缩包, 内部用 XML/XHTML 描述元数据与正文。

解析流程 (EPUB 2 / EPUB 3 通吃):
  1. 读 META-INF/container.xml  -> 找到 OPF(package) 文件路径
  2. 解析 OPF -> metadata(书名/作者) + manifest(资源清单) + spine(阅读顺序)
  3. 按 spine 顺序取出每个 XHTML -> 剥离标签得到纯文本

用法:
  python epubParsor.py 一本书.epub              # 元数据 + 章节目录
  python epubParsor.py 一本书.epub --text       # 打印全书纯文本
  python epubParsor.py 一本书.epub --chapter 2  # 只看第 2 章
  python epubParsor.py 一本书.epub -o out.txt   # 导出纯文本, 可再喂给 vector_demo.py --ingest

当模块用:
  from epubParsor import EpubBook
  with EpubBook("一本书.epub") as book:
      print(book.title, book.author)
      for ch in book.chapters():
          print(ch.index, ch.title, len(ch.text))
"""

import re
import sys
import posixpath
import argparse
import zipfile
import xml.etree.ElementTree as ET
from urllib.parse import unquote
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

CONTAINER_PATH = "META-INF/container.xml"

# 这些标签只贡献换行, 不产生字符
_BLOCK_TAGS = {
    "p", "div", "br", "hr", "li", "tr", "td", "th",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "section", "article", "aside", "header", "footer",
    "blockquote", "pre", "figcaption", "dd", "dt", "table",
}
# 这些整块内容直接丢弃(脚本/样式/文档头)
_DROP_RE = re.compile(r"<(head|script|style)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_TITLE_RE = re.compile(r"<h[1-3]\b[^>]*>(.*?)</h[1-3]>", re.IGNORECASE | re.DOTALL)
_DOC_TITLE_RE = re.compile(r"<title\b[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)

def _local_name(tag):
    """剥掉 XML 命名空间, 只留标签本地名 (如 '{...}title' -> 'title')。"""
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""

def decode_bytes(raw):
    """把字节还原成字符串: 先看 BOM, 再试 utf-8, 最后交给 charset_normalizer。"""
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:
        from charset_normalizer import from_bytes

        best = from_bytes(raw).best()
        if best is not None:
            return str(best)
    except Exception:
        pass
    return raw.decode("utf-8", errors="replace")

class _HtmlToText(HTMLParser):
    """把 XHTML 片段转成纯文本: 丢掉标签, 保留段落换行。"""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._parts = []

    def handle_starttag(self, tag, attrs):
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_startendtag(self, tag, attrs):
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data):
        self._parts.append(data)

    def text(self):
        raw = "".join(self._parts)
        lines = (re.sub(r"[ \t\u00a0]+", " ", ln).strip() for ln in raw.splitlines())
        return "\n".join(ln for ln in lines if ln)

def html_to_text(html):
    """HTML/XHTML -> 纯文本 (先丢弃 head/script/style, 再剥标签)。"""
    html = _DROP_RE.sub(" ", html)
    parser = _HtmlToText()
    parser.feed(html)
    parser.close()
    return parser.text()

@dataclass
class Chapter:
    index: int      # 从 1 开始的章号
    id: str         # OPF 里 item 的 id
    href: str       # 相对 OPF 的文件路径
    title: str      # 猜测的章节标题
    text: str       # 纯文本正文

class EpubBook:
    """一个可 with 使用的 EPUB 阅读器。"""

    def __init__(self, path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"文件不存在: {self.path}")
        if not zipfile.is_zipfile(self.path):
            raise ValueError(f"不是有效的 EPUB(ZIP) 文件: {self.path}")
        self._zip = zipfile.ZipFile(self.path)
        self._names = set(self._zip.namelist())
        self.opf_path = self._find_opf()
        self.opf_dir = posixpath.dirname(self.opf_path)
        self.metadata = {}
        self.manifest = {}   # id -> {"href", "media_type", "path"}
        self.spine = []      # 按阅读顺序排列的 item id
        self._parse_opf()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._zip.close()

    # ---------- 元数据快捷方式 ----------
    @property
    def title(self):
        return self.metadata.get("title") or self.path.stem

    @property
    def author(self):
        return self.metadata.get("creator", "未知")

    @property
    def language(self):
        return self.metadata.get("language", "")

    # ---------- 内部: 定位与解析 OPF ----------
    def _find_opf(self):
        try:
            data = self._zip.read(CONTAINER_PATH)
        except KeyError:
            raise ValueError("不是有效的 EPUB: 缺少 META-INF/container.xml")
        root = ET.fromstring(data)
        for el in root.iter():
            if _local_name(el.tag) == "rootfile":
                full = el.get("full-path")
                if full:
                    return full
        raise ValueError("container.xml 中没有 rootfile(full-path)")

    def _resolve(self, href):
        """把 manifest 里的相对 href 解析成 zip 内的绝对路径。"""
        href = href.split("#", 1)[0]
        return posixpath.normpath(posixpath.join(self.opf_dir, href)).lstrip("/")

    def _read_member(self, path):
        for cand in (path, unquote(path)):
            if cand in self._names:
                return self._zip.read(cand)
        raise KeyError(path)

    def _parse_opf(self):
        root = ET.fromstring(self._zip.read(self.opf_path))
        meta_tags = {"title", "creator", "language", "publisher", "date",
                     "identifier", "description", "subject"}
        for el in root.iter():
            name = _local_name(el.tag)
            if name == "item":
                item_id, href = el.get("id"), el.get("href")
                if item_id and href:
                    self.manifest[item_id] = {
                        "href": href,
                        "media_type": el.get("media-type", ""),
                        "path": self._resolve(href),
                    }
            elif name == "itemref":
                idref = el.get("idref")
                if idref:
                    self.spine.append(idref)
            elif name in meta_tags and el.text and el.text.strip():
                value = el.text.strip()
                old = self.metadata.get(name)
                if old and value not in old:
                    self.metadata[name] = f"{old}, {value}"
                else:
                    self.metadata.setdefault(name, value)

    # ---------- 章节 ----------
    def _guess_title(self, html, fallback):
        for regex in (_TITLE_RE, _DOC_TITLE_RE):
            m = regex.search(html)
            if m:
                t = html_to_text(m.group(1))
                if t:
                    return t
        return f"第 {fallback} 章"

    def chapters(self):
        """按 spine 顺序逐章产出 Chapter。"""
        index = 0
        for idref in self.spine:
            info = self.manifest.get(idref)
            if not info:
                continue
            media = info["media_type"]
            href_l = info["href"].lower()
            is_html = media in ("application/xhtml+xml", "text/html") or (
                not media and href_l.endswith((".xhtml", ".html", ".htm"))
            )
            if not is_html:
                continue
            try:
                html = decode_bytes(self._read_member(info["path"]))
            except KeyError:
                continue
            index += 1
            yield Chapter(
                index=index,
                id=idref,
                href=info["href"],
                title=self._guess_title(html, index),
                text=html_to_text(html),
            )

    def full_text(self, with_title=True):
        """全书纯文本; with_title=True 时在每章前插入章名。"""
        parts = []
        for ch in self.chapters():
            if with_title:
                parts.append(ch.title)
            parts.append(ch.text)
        return "\n\n".join(p for p in parts if p)

def print_book_info(book):
    print(f"📖 {book.title}")
    print(f"   作者: {book.author}")
    if book.language:
        print(f"   语言: {book.language}")
    if book.metadata.get("publisher"):
        print(f"   出版: {book.metadata['publisher']}")
    print(f"   OPF : {book.opf_path}")
    print(f"   资源: {len(book.manifest)} 项, spine: {len(book.spine)} 项")
    print("   ---------- 章节目录 ----------")
    for ch in book.chapters():
        print(f"   {ch.index:>3}. {ch.title}   ({len(ch.text)} 字)")

def main():
    p = argparse.ArgumentParser(description="EPUB 解析器 (纯标准库, 提取元数据/目录/正文)")
    p.add_argument("epub", help="要解析的 .epub 文件路径")
    p.add_argument("-t", "--text", action="store_true", help="打印全书纯文本")
    p.add_argument("-c", "--chapter", type=int, help="只打印第 N 章正文(从 1 开始)")
    p.add_argument("-o", "--out", metavar="FILE", help="把纯文本写入文件(可再喂给 vector_demo.py --ingest)")
    args = p.parse_args()

    try:
        book = EpubBook(args.epub)
    except Exception as e:
        sys.exit(f"✗ 解析失败: {e}")

    with book:
        if args.out:
            Path(args.out).write_text(book.full_text(), encoding="utf-8")
            print(f"✓ 已导出纯文本到 {Path(args.out).resolve()}")
            return

        if args.chapter:
            for ch in book.chapters():
                if ch.index == args.chapter:
                    print(ch.text)
                    return
            sys.exit(f"✗ 没有第 {args.chapter} 章")

        if args.text:
            print(book.full_text())
            return

        print_book_info(book)

if __name__ == "__main__":
    main()