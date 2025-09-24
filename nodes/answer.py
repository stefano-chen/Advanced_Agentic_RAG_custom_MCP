from utils.state import AgentState
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.prompts import PromptTemplate

class GenerateAnswer:
    """
    The Answer Generation Node
    """
    def __init__(self, llm: BaseChatModel, prompt: str):
        """
        Attributes:
            llm (BaseChatModel): the LLM tasked with the answer generation
            prompt (str): the prompt used for the answer generation
        """
        self._prompt = prompt
        self._llm = llm

    def generate_answer(self, state: AgentState) -> AgentState:
        """
        Use the original question and the context to generate a answer

        Parameters:
            state (AgentState): the graph state

        Returns:
            AgentState: the updated graph state
        """
        question = state["original_question"]
        context = state["context"]
        prompt = PromptTemplate.from_template(self._prompt).invoke({"question": question, "context": context})

        response = self._llm.invoke(prompt)

        state["messages"].append(response)

        return state
        
        