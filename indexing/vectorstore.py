from langchain_core.embeddings import Embeddings
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_community.docstore.in_memory import InMemoryDocstore
from pathlib import Path
import faiss

class VectorStore:
    
    def __init__(self, store_type: str, embedding_model: Embeddings, save_dir_path: str = None):
        self.store_type = store_type
        self.save_dir_path = save_dir_path
        self.embedding_model = embedding_model

        # Use the chosen vector store type
        if self.store_type == "faiss":
            self.vectorstore = FAISS(
                embedding_function=self.embedding_model,    
                index=faiss.IndexFlatL2(len(self.embedding_model.embed_query(" "))),
                docstore=InMemoryDocstore(),
                index_to_docstore_id={}
            )
        elif self.store_type == "chroma":
            self.vectorstore = Chroma(
                embedding_function=self.embedding_model,
                persist_directory=self.save_dir_path
            )
        else:
            raise NotImplementedError(f"Vector store {self.store_type} is not supported")
    
    def as_retriever(self, k: int):
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    def add_documents(self, documents: List[Document]):
        if not documents:
            raise ValueError("None or empty value found")
        return self.vectorstore.add_documents(documents=documents)
    
    def load(self):
        # Note: When using ChromaDB you are not required to use the load method
        # Just pass the loading directory to the __init__

        dir = Path(self.save_dir_path)
        if not dir.exists():
            raise FileNotFoundError(f"Directory {dir} not found")
        
        if self.store_type == "faiss":
            self.vectorstore = FAISS.load_local(
                folder_path=self.save_dir_path,
                embeddings=self.embedding_model,
                allow_dangerous_deserialization=True
            )

    def save(self):
        # Note: When using ChromaDB you are not required to use the save method
        # Just pass the saving directory to the __init__
        if self.store_type == "chroma":
            return
        
        if self.save_dir_path:
            dir = Path(self.save_dir_path)
            if not dir.exists():
                dir.mkdir(parents=True)
            
            if self.store_type == "faiss":
                self.vectorstore.save_local(folder_path=dir)
        else:
            raise Exception("save path not provided in the instantiation phase")

    