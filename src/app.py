"""Application for a Q&A chatbot using uploaded documents"""

import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import sqlite3
import streamlit as st
from data_loader import document_handler
from file_parser import build_index, vector_store, clear_vector_store
from retrieval import retrieve_similar_docs
from llm import final_answer

st.set_page_config(page_title="Data Chat📚", layout="wide")
st.markdown(
    "<h1 style='text-align: center; "
    "font-size: 3em; "
    "font-family: Times New Roman, Times, serif; "
    "font-weight: bold;'>Q&amp;A with Your Documents 📖</h1>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Document Management")
    uploaded_docs = st.file_uploader(
        label="Upload your documents", accept_multiple_files=True
    )
    if st.button("Process Documents", use_container_width=True):
        if not uploaded_docs:
            st.warning("Please upload one or more files first.")
        else:
            with st.spinner("Processing documents..."):
                clear_vector_store(vector_store)
                for file in uploaded_docs:
                    processed = document_handler(file)
                    build_index(processed)
                st.success("Documents processed and indexed!")
    if st.button("🧹 Clear Chat", use_container_width=True):
        if len(st.session_state.get("messages", [])) <= 1:
            st.info("Begin chatting to remove the evidence :)")
        else:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Hi! I am your AI assistant. Upload documents to begin chatting.",
                }
            ]
            st.success("Evidence cleared! Start a new chat.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I am your AI assistant. Upload documents to begin chatting.",
            "avatar": "🥸",
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar", "🥸")):
        st.write(msg["content"])

if prompt := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display the user's message immediately
    with st.chat_message("user", avatar="🥹"):
        st.write(prompt)
    with st.chat_message("assistant", avatar="🥸"):
        with st.spinner("Thinking..."):
            similar_chunks = retrieve_similar_docs(prompt)
            if similar_chunks:
                answer = final_answer(similar_chunks, prompt)
                st.write(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            else:
                st.warning(
                    "No similar chunks found; try re-uploading or refining your query."
                )
