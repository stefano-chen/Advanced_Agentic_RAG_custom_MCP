from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.embeddings import Embeddings

class EmbeddingModel:
    """
    A class to manage the Embedding Model selection process based on the configuration file
    """

    def __init__(self, config: dict[str, str]):
        """
        Instantiate the right Embedding model based on the configuration file

        Parameters:
            config (Dict[str, str]): the configuration file for the Embedding Model
        Raises:
            NotImplementedError: when the configuration file contains a not supported Embedding provider

        """
        if config["embedding_provider"] == "huggingface":
            self.embedding_model = HuggingFaceEmbeddings(model_name=config["embedding_model"])
        elif config["embedding_provider"] == "openai":
            self.embedding_model = OpenAIEmbeddings(model=config["embedding_model"])
        elif config["embedding_provider"] == "ollama":
            self.embedding_model = OllamaEmbeddings(model=config["embedding_model"])
        elif config["embedding_provider"] == "google":
            self.embedding_model = GoogleGenerativeAIEmbeddings(model=config["embedding_model"])
        else:
            raise NotImplementedError(f"Embedding provider {config["embedding_provider"]} not supported")

    def get(self) -> Embeddings:
        """
        Returns:
            Embeddings: the selected Embedding model
        """
        return self.embedding_model