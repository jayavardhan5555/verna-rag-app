import os
import tempfile
from pathlib import Path

from langchain_community.document_loaders import (WebBaseLoader, PyPDFLoader, Docx2txtLoader)

def load_documents(source_tye:str, source) -> list:
    """Load docsuments from the given source type and source."""

    if source_tye == "file":
        suffix = Path(source.name).suffix.lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(source.read())
            tmp_path = tmp.name
        
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix in {".docx", ".doc"}: 
            loader = Docx2txtLoader(tmp_path)
        else:
            raise ValueError(f"Unsupported file type: {source.type}")
        
        docs = loader.load()
        os.unlink(tmp_path)
    
    elif source_tye == "url":
        loader = WebBaseLoader(source)
        docs = loader.load()
    
    else:
        raise ValueError(f"Unsupported source type: {source_tye}")

    return docs