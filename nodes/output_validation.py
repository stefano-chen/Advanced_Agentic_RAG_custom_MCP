from langchain_core.language_models.chat_models import BaseChatModel
from utils.state import AgentState
from langchain.prompts import PromptTemplate

class AnswerValidation:
    """
    The Answer Validation Node
    """
    def __init__(self, llm: BaseChatModel, prompt: str):
        """
        Attributes:
            llm (BaseChatModel): the LLM for answer validation
            prompt (str): the prompt used by the LLM
        """
        self._llm = llm
        self._prompt = prompt

    def validate(self, state: AgentState) -> AgentState:
        """
        check for hallucinations in the previous message

        Parameters:
            state (AgentState): the graph state

        Returns:
            AgentState: the updated graph state
        """
        question = state['original_question']
        context = state['context']
        answer = state['messages'][-1].content
        prompt = PromptTemplate.from_template(self._prompt).invoke({"question":question, "context": context, "answer": answer})
        response = self._llm.invoke(prompt)

        if "pass" not in response.content.lower():
            state['messages'].append(response)

        return state
    

