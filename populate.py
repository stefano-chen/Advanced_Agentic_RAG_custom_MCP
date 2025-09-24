import json
import os
from indexing.document_loader import DocumentLoader
from indexing.chunking import Chunking
from indexing.vectorstore import VectorStore
from utils.embedding import EmbeddingModel
from pathlib import Path
from dotenv import load_dotenv

load_dotenv("./.env")

if __name__ == "__main__":

    # Read the configuration json file
    if not Path("./config/populate_config.json").exists():
        raise FileNotFoundError("populate_config.json not Found")

    with open("./config/populate_config.json", "r") as f:
        config = json.load(f)

    # Instantiate the chosen Embedding model
    embedding_model = EmbeddingModel(config["embedding"]).get()
    
    data_dir_path = config["data_dir_path"]
    if not os.path.exists(data_dir_path):
        raise NotADirectoryError(f"{data_dir_path} is not a directory")
    
    for entry in os.scandir(data_dir_path):
        if entry.is_file():
            # load Documents
            print(f"loading from {entry.path}")
            pages = DocumentLoader(file_type=entry.path.split(".")[-1], file_path=entry.path).load()
            print(f"{len(pages)} pages loaded")
            # Apply chunking strategy
            print(f"Apply chunking strategy {config["chunking_strategy"]}")
            chunks = Chunking(config["chunking_strategy"], config["chunking_options"], embedding_model).apply(pages)
            print(f"{len(chunks)} chunks extracted")
            # Create vector store for each document (just to demonstrate the llm capability to choose the right vector store)
            print(f"Creating Vector store {config["vector_store_type"]}")
            save_dir = Path(config["save_dir_path"]).joinpath(entry.name.split(".")[0])
            vector_store = VectorStore(config["vector_store_type"], embedding_model, save_dir)
            vector_store.add_documents(chunks)
            # Persist the vector store, making it accessible by the agent application
            print(f"saving vector store at {save_dir}\n")
            vector_store.save()
