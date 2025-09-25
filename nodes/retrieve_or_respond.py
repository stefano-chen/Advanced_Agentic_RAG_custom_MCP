from langchain_core.language_models.chat_models import BaseChatModel
from typing import List
from langchain.prompts import PromptTemplate
from utils.state import AgentState
from langchain_core.messages import AnyMessage

def _get_past_tool_calls(messages: List[AnyMessage]) -> str:
    """
    Extract all and only tool calls from the list of messages
    to produce a string representing the "past tool calls"

    Parameters:
        messages (List[AnyMessage]): a list of messages 

    Returns:
        str: a string representation of the past tool calls
    """
    past_tool_calls = ""
    for msg in messages:
        if msg.type == "ai" and hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0:
            past_tool_calls += str(msg.tool_calls) + "\n"
    return past_tool_calls


class Retrieve_Respond:
    """
    The Retrieve or Respond Node
    """
    def __init__(self, llm: BaseChatModel, prompt: str):
        """
        Attributes:
            llm (BaseChatModel): the LLM for this node
            prompt (str): the prompt used to validate the user's query
        """
        self._llm = llm
        self._prompt = prompt
    
    def choose(self, state: AgentState) -> AgentState:
        """
        This method define the retrieve or respond choise process

        Parameters:
            state (AgentState): the graph state

        Returns:
            AgentState: the update graph state
        """
        question = state["original_question"]
        context = state["context"]
        past_tool_calls = _get_past_tool_calls(state["messages"])
        prompt = PromptTemplate.from_template(self._prompt).invoke({"question": question, "context": context, "past_tool_calls": past_tool_calls})
        response = self._llm.invoke(prompt)
        state["messages"].append(response)
        return state