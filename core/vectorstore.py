"""Chroma Vectore Store Creation from Langchain."""

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP

def build_vector_store(documents: list, api_key : str) -> Chroma:
    """Splits docs into chunks , emded them and retun in memory Chorma store"""

    splitter = RecursiveCharacterTextSplitter(chunk_size = CHUNK_SIZE, chunk_overlap = CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("No chunks were created from the documents. Please check the input documents.")
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return Chroma.from_documents(chunks, embeddings)