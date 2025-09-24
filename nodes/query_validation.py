from langchain_core.language_models.chat_models import BaseChatModel
from typing import List, Literal
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from utils.state import AgentState

class QueryValidation:
    """
    The Query Validation Node
    """
    def __init__(self, llm: BaseChatModel, prompt: str, topics: List[str]):
        """
        Attributes:
            llm (BaseChatModel): the LLM for this node
            prompt (str): the prompt used to validate the user's query
            topics (List[str]): a list of accepted topics
        """
        self._llm = llm
        self._prompt = prompt
        self._topics = topics

    def validate(self, state: AgentState) -> AgentState:
        """
        Checks if the user's query is related to provided topics

        Parameters:
            state (AgentState): the graph state

        Returns:
            AgentState: the updated graph state
        """
        question = state["original_question"]
        prompt_template = PromptTemplate.from_template(self._prompt)
        prompt = prompt_template.invoke({"question": question, "topics": self._topics})
        response = self._llm.invoke(prompt)
        state["messages"].append(AIMessage(f"Is \"{question}\" related with at least one of this topics {self._topics}? {response.content}"))
        return state
        

def is_related(state: AgentState) -> Literal["yes", "no"]:
    """
    This function is used for the conditional edge.

    Parameters:
        state (AgentState): the graph state after the query validation node

    Returns:
        (Literal["yes", "no"]): a string used by the graph to determine the next node to reach
    """
    last_msg = state["messages"][-1]
    if "yes" in last_msg.content.lower():
        return "yes"
    return "no"