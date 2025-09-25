from langchain_core.language_models.chat_models import BaseChatModel
from typing import List, Callable, Union, Literal
from langchain_core.tools import BaseTool
from langchain.prompts import PromptTemplate
from utils.state import AgentState
from langchain_core.messages import AnyMessage, AIMessage

class ToolRouting:
    """
    The Tool Routing Node
    """
    def __init__(self, llm: BaseChatModel, prompt: str, tools: List[Union[Callable, BaseTool]]):
        """
        Attributes:
            llm (BaseChatModel): the LLM for this node
            prompt (str): the prompt used to validate the user's query
            tools (List[Union[Callable, BaseTool]]): a list of available tools
        """
        # This let the LLM know which tools are available to call
        self._llm = llm.bind_tools(tools)
        self._prompt = prompt

    def route(self, state: AgentState) -> AgentState:
        """
        This method check the last message and based on it decide which tool to call

        Parameters:
            state (AgentState): the graph state
        
        Returns:
            AgentState: the updated graph state
        """
        last_msg = state["messages"][-1]
        past_tool_calls = _get_past_tool_calls(state["messages"])

        if "retrieve" in last_msg.content.lower():
            prompt = PromptTemplate.from_template(self._prompt).invoke({"tools": past_tool_calls, "query": state["question"]})
            response = self._llm.invoke(prompt)
            state["messages"].append(response)
        else: 
            state["messages"].append(AIMessage("No tools to call, routing to the generate answer node"))
        return state


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


def tool_condition(state: AgentState) -> Literal["retrieve", "respond"]:
    """
    check if the last message is a tool call.
    
    Parameters:
        state (AgentState): The graph state

    Returns:
        str: a string with value "retrieve" if the last message contains a tool call, otherwise has value "respond"

    """
    last_msg = state['messages'][-1]

    if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0:
        return "retrieve"
    return "respond"