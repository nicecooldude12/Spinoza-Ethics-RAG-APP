# ingest.py

import os
import shutil

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from database import get_vector_store


CHROMA_PATH = "chroma_db"
DATA_FILE = "data/spinoza.txt"


def load_text_file(file_path):
    """
    Loads a plain text file and wraps it as a LangChain Document.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find file: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    print("Raw document length:", len(text))

    if not text.strip():
        raise ValueError(f"The file {file_path} is empty.")

    document = Document(
        page_content=text,
        metadata={"source": file_path}
    )

    return [document]


def split_documents(documents):
    """
    Splits large documents into smaller chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    print("Number of chunks created:", len(chunks))

    if len(chunks) == 0:
        raise ValueError("No chunks were created. Check your source document.")

    return chunks


def save_to_chroma(chunks):
    """
    Stores document chunks in Chroma.
    """

    if len(chunks) == 0:
        raise ValueError("Cannot save to Chroma because chunks list is empty.")

    vector_store = get_vector_store()

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    print("Number of IDs created:", len(ids))
    print("First chunk preview:", chunks[0].page_content[:100])

    vector_store.add_documents(
        documents=chunks,
        ids=ids
    )

    print(f"Saved {len(chunks)} chunks to Chroma database.")


def main():
    # Optional: reset the database every time while testing
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Old Chroma database deleted.")

    documents = load_text_file(DATA_FILE)
    chunks = split_documents(documents)
    save_to_chroma(chunks)


if __name__ == "__main__":
    main()