from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
from bs4 import BeautifulSoup
from ddgs import DDGS

from ..service import DistributedRAGService


def _search_web(query: str, limit: int = 5) -> List[Dict[str, bytes]]:
    values: List[Dict[str, bytes]] = []
    with DDGS() as client:
        results = list(client.text(query=query, region="wt-wt", safesearch="moderate", max_results=limit))
    for index, result in enumerate(results):
        url = result.get("href")
        if not url:
            continue
        try:
            response = requests.get(
                url,
                timeout=10,
                headers={"User-Agent": "DistributedRAG/2.0"},
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            for tag in soup(["script", "style", "nav"]):
                tag.decompose()
            text = re.sub(r"\s+", " ", soup.get_text(" ")).strip()
            if text:
                values.append({"name": f"web-{index + 1}.txt", "content": text.encode("utf-8"), "url": url})
        except requests.RequestException:
            continue
    return values


def run_streamlit_app(profile: Optional[str] = None) -> None:
    st.set_page_config(page_title="DistributedRAG", layout="wide")
    st.title("🚀 DistributedRAG 分布式知识库")
    st.markdown("上传多源文档并建立持久化知识库，然后基于可追溯证据进行问答。")

    @st.cache_resource
    def service(selected_profile: str) -> DistributedRAGService:
        return DistributedRAGService(profile=selected_profile)

    rag = service(profile)
    st.session_state.setdefault("document_ids", [])
    st.session_state.setdefault("jobs", [])
    st.session_state.setdefault("response", None)

    with st.sidebar:
        st.subheader("⚙️ 检索设置")
        use_hyde = st.toggle("启用 HyDE 补充召回", value=rag.config.retrieval.use_hyde)
        use_web = st.toggle("加入联网搜索结果", value=False)
        if st.button("服务健康检查"):
            st.json(rag.health())
        if st.session_state.jobs:
            st.subheader("最近任务")
            for job in st.session_state.jobs[-5:]:
                st.caption(f"{job['job_id']} · {job['status']}")

    with st.form("rag-form"):
        uploaded_files = st.file_uploader(
            "上传知识库文件",
            accept_multiple_files=True,
            type=["pdf", "doc", "docx", "ppt", "pptx", "xls", "xlsx", "csv", "md", "txt", "html", "png", "jpg", "jpeg", "tif", "tiff", "wav", "mp3", "m4a", "flac"],
        )
        query = st.text_input("问题", placeholder="例如：文档的主要结论是什么？")
        submitted = st.form_submit_button("摄取文档并提问")

    if submitted:
        if not query:
            st.error("请输入问题。")
            return
        try:
            inputs: List[Dict[str, Any]] = [
                {"name": file.name, "content": file.getvalue(), "metadata": {"content_type": file.type or ""}}
                for file in uploaded_files or []
            ]
            if use_web:
                with st.spinner("正在获取联网资料……"):
                    inputs.extend(
                        {"name": item["name"], "content": item["content"], "metadata": {"source_url": item["url"]}}
                        for item in _search_web(query)
                    )
            if inputs:
                with st.spinner("正在分布式解析、向量化并发布文档版本……"):
                    for item in inputs:
                        job = rag.ingest(item["name"], item["content"], item["metadata"])
                        st.session_state.jobs.append(job)
                        document_id = job["document_id"]
                        if document_id not in st.session_state.document_ids:
                            st.session_state.document_ids.append(document_id)
            if not st.session_state.document_ids:
                st.warning("请先上传至少一个文档。")
                return
            with st.spinner("正在检索、重排和生成带引用的回答……"):
                st.session_state.response = rag.ask(
                    query,
                    document_ids=st.session_state.document_ids,
                    use_hyde=use_hyde,
                )
        except Exception as exc:
            st.error(f"处理失败：{exc}")

    response = st.session_state.response
    if response:
        st.subheader("回答")
        st.write(response["answer"])
        st.caption(f"Trace ID: {response['trace_id']}")
        if response["citations"]:
            st.subheader("引用来源")
            for citation in response["citations"]:
                locator = {key: value for key, value in citation["source_locator"].items() if value not in (None, [], "")}
                st.info(f"[{citation['source_id']}] {citation['claim']}\n\n定位：{locator}")
        elif not response["evidence_sufficient"]:
            st.warning("当前证据不足。")
