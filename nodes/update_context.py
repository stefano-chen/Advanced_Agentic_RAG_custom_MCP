from utils.state import AgentState
from langchain_core.messages import AIMessage

def update_context(state: AgentState) -> AgentState:
    """
    Merge chunks into a single context string

    Parameters:
        state (AgentState): the graph state

    Returns:
        AgentState: the updated graph state
    """
    new_context = ""

    message = None

    if state["chunks"]:
        new_context = "\n\n".join(state['chunks'])
        message = AIMessage("Context Updated")
        state["chunks"] = None
        state['context'] = state['context'] + new_context
    else:
        message = AIMessage("No new Context")

    state['messages'].append(message)

    return state