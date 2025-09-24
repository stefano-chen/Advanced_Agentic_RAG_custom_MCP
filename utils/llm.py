from langchain_openai.chat_models import ChatOpenAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_ollama.chat_models import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel
from typing import Dict, Any

class LLMModel:
    """
    A class to manage the LLM selection process based on the configuration file
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Instantiate the right LLM model based on the configuration file

        Parameters:
            config (Dict[str, Any]): the configuration file for the LLM
        Raises:
            NotImplementedError: when the configuration file contains a not supported LLM provider

        """
        if config["llm_provider"] == "lm-studio":
            self._llm = ChatOpenAI(
                name=config["llm_model"], 
                base_url=config["llm_host"], 
                api_key="not needed", 
                temperature=config["temperature"]
            )
        elif config["llm_provider"] == "google":
            self._llm = ChatGoogleGenerativeAI(
                model=config["llm_model"], 
                temperature=config["temperature"]
            )
        elif config["llm_provider"] == "openai":
            self._llm = ChatOpenAI(
                name=config["llm_model"], 
                base_url=config["llm_host"], 
                temperature=config["temperature"]
            )
        elif config["llm_provider"] == "ollama":
            self._llm = ChatOllama(
                model=config["llm_model"], 
                temperature=config["temperature"]
            )
        else:
            raise NotImplementedError(f"LLM provider {config["llm_provider"]} not supported")
        
    def get(self) -> BaseChatModel:
        """
        Returns:
            BaseChatModel: the selected LLM
        """
        return self._llm