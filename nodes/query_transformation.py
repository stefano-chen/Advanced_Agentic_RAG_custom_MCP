
from utils.state import AgentState
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from typing import Dict, Any

class QueryTransform:
    """
    The Query Transformation Node
    """
    def __init__(self, strategy: str, options: Dict[str, Dict[str, Any]], llm: BaseChatModel, prompts: Dict[str, str]):
        """
        Attributes:
            stategy (str): the name of a query transformation technique (e.g. step-back, hyde, etc.)
            options: (Dict[str, Dict[str, Any]]): a dictionary that for each technique, contains the respective parameters
            llm (BaseChatModel): the LLM for this node
            options: (Dict[str, str]): a dictionary that for each technique, contains the respective prompt
        
        Raises:
            NotImplementedError: if the provided strategy is not supported
        """
        self._llm = llm

        if strategy == "step-back":
            self._prompt = prompts.get("step-back", None)
            strategy_options = options.get("step-back")
            if not strategy_options:
                raise Exception(f"{strategy} options not found")
            self._max_char = strategy_options.get("max_char", 100)
        elif strategy == "hyde":
            strategy_options = options.get("hyde")
            if not strategy_options:
                raise Exception(f"{strategy} options not found")
            self._prompt = prompts.get("hyde", None)
            self._max_char = options.get("max_char", 500)
        else:
            raise NotImplementedError(f"Query transformation {strategy} is not supported")

    
    def transform(self, state: AgentState) -> AgentState:
        """
        Rewrite the user's query using query transformation technique

        Parameters:
            state (AgentState): the graph state

        Returns:
            AgentState: the updated graph state
        """
        if not self._prompt:
            raise Exception("Query transformation prompt not Found")
        
        question = state["original_question"]

        prompt = PromptTemplate.from_template(self._prompt).invoke({"max_char": self._max_char, "question": question})

        rewritten_question = self._llm.invoke(prompt).content.strip()
        state["messages"].append(AIMessage(f"\"{question}\" -> {rewritten_question}"))
        state["question"] = rewritten_question
        return state
        
