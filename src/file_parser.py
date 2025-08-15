"""Data loader module for handling various document types"""

import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
)

splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=400)

temp_db = tempfile.mkdtemp()
vector_store = Chroma(embedding_function=embedding_model, persist_directory=temp_db)


def build_index(combined_doc):
    """Build an index from the combined documents and store them in the vector store"""
    chunks = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(splitter.split_documents, [doc]): doc
            for doc in combined_doc
        }
        for f in as_completed(futures):
            chunks.extend(f.result())

    vector_store.add_documents(chunks)
    print(f"Indexed {len(chunks)} chunks into Storage")


def clear_vector_store(store):
    """Clear the vector store by deleting all indexed documents"""
    all_ids = store.get()["ids"]
    if all_ids:
        store.delete(ids=all_ids)
        print(f"Deleted {len(all_ids)} documents from the vector store.")
    else:
        print("No documents to delete.")
