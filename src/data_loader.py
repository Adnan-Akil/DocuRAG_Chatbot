"""Data loader module for handling various document types"""

import os
import tempfile

import pandas as pd
from docx import Document as docxLoader
from langchain.docstore.document import Document as LangDoc
from langchain_community.document_loaders import PyMuPDFLoader


def read_pdf(file):
    """Read PDF file and return its content as LangDoc objects"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file.read())
        temp_path = temp_file.name

    try:
        loader = PyMuPDFLoader(temp_path)
        documents = loader.load()
        cleaned_text = " ".join(
            doc.page_content.replace("-\n", "").replace("\n", " ") for doc in documents
        )

        return [LangDoc(page_content=cleaned_text, metadata={"source": file.name})]
    finally:
        os.remove(temp_path)


def read_docx_file(file):
    """Read DOCX file and return its content as LangDoc objects"""
    doc = docxLoader(file)
    text = "\n".join([p.text for p in doc.paragraphs])
    return [LangDoc(page_content=text, metadata={"source": file.name})]


def read_txt(file):
    """Read TXT file and return its content as LangDoc objects"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(file.read())
        temp_path = temp_file.name

    try:
        with open(temp_path, "r", encoding="utf-8") as f:
            content = f.read()
        return [LangDoc(page_content=content, metadata={"source": file.name})]
    finally:
        os.remove(temp_path)


def read_excel(file):
    """Read Excel file and return its content as LangDoc objects"""
    doc = pd.read_excel(file)
    content = doc.to_string(index=False)
    return [LangDoc(page_content=content, metadata={"source": file.name})]


def read_csv(file):
    """Read CSV file and return its content as LangDoc objects"""
    doc = pd.read_csv(file)
    content = doc.to_string(index=False)
    return [LangDoc(page_content=content, metadata={"source": file.name})]


def document_handler(uploaded_file):
    """Handle different document types and return their content as LangDoc objects"""
    path_type = os.path.splitext(uploaded_file.name)[1].lower()
    print(path_type)

    if path_type == ".pdf":
        return read_pdf(uploaded_file)
    if path_type == ".docx":
        return read_docx_file(uploaded_file)
    if path_type == ".xlsx":
        return read_excel(uploaded_file)
    if path_type == ".csv":
        return read_csv(uploaded_file)
    if path_type == ".txt":
        return read_txt(uploaded_file)
