from typing import Dict, Any, List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker


class Chunking:
    def __init__(self, chunking_strategy: str, chunking_options: Dict[str, Any], embedding_model: Embeddings = None):
        self.chunking_strategy = chunking_strategy

        if self.chunking_strategy.lower() == "window":
            self.text_splitter = RecursiveCharacterTextSplitter(**chunking_options)
        elif self.chunking_strategy.lower() == "sentence":
            # Split on . ? !
            self.text_splitter = CharacterTextSplitter(separator=r'(?<=[.!?])\s+', is_separator_regex=True, **chunking_options)
        elif self.chunking_strategy.lower() == "paragraph":
            # Simple paragraph split
            self.text_splitter = CharacterTextSplitter(separator="\n\n", **chunking_options)
        elif self.chunking_strategy.lower() == "semantic":
            if embedding_model:
                self.text_splitter = SemanticChunker(embeddings=embedding_model)
            else:
                raise Exception("embedding model is required when using semantic chunking")
        else:
            raise NotImplementedError(f"Chunking strategy {self.chunking_strategy} not supported")
        
    def apply(self, documents: List[Document]):
        if not documents:
            raise ValueError("None or empty value found")
        
        return self.text_splitter.split_documents(documents=documents)
