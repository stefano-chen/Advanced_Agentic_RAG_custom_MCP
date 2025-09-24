from utils.state import AgentState
from typing import List
from langchain.prompts import PromptTemplate
import numpy as np
from langchain_core.messages import AIMessage
from utils.llm import LLMModel
from utils.embedding import EmbeddingModel
from typing import Dict

class Reranking:
    """
    The Reranking Node
    """
    def __init__(self, strategies: List[str], weights: List[float], options: Dict[str, Dict[str, str]], prompts: Dict[str, str]):
        """
        Attributes:
            stategies (List[str]): a list of string representing the reranking strategies to apply (e.g. ["semantic", "distance"])
            weights (List[float]): a list of float representing the weight for each strategy. **It must sum up to 1**
            options: (Dict[str, Dict[str, str]]): a dictionary that for each stratety, contains the respective parameters
            prompts: (Dict[str, str]): a dictionary containing the prompts
        """
        self._options = options
        self._prompts = prompts
        self._strategies = strategies
        self._weights = weights

    def _calculate_semantic_score(self, question: str, chunks: List[str]) -> List[float]:
        """
        For each chunk, it calculates the semantic score for a provided question

        Parameters:
            question (str): the question used to calculate the score
            chunks (List[str]): a list of chunks to evaluate
        
        Returns:
            List[float]: the calculated scores
        """
        scores = []
        prompt_template = self._prompts.get("reranking")
        if not prompt_template:
            raise Exception("Reranking prompt not Found")
        
        llm_config = self._options.get("semantic")

        if not llm_config:
            raise Exception("Semantic llm options not Found")

        llm = LLMModel(llm_config).get()

        for chunk in chunks:
            prompt = PromptTemplate.from_template(prompt_template).invoke({"question": question, "chunk": chunk})
            grade = llm.invoke(prompt).content
            try:
                scores.append(float(grade))
            except:
                scores.append(0)
        
        return scores
    
    def _calculate_distance_score(self, question: str, chunks: List[str]):
        """
        For each chunk, it calculates the distance score from a provided question

        Parameters:
            question (str): the question used to calculate the score
            chunks (List[str]): a list of chunks to evaluate
        
        Returns:
            List[float]: the calculated scores
        """
        embedding_config = self._options.get("distance")

        if not embedding_config:
            raise Exception("Distance embedding options not Found")

        embedding_model = EmbeddingModel(embedding_config).get()

        question_emb = np.array(embedding_model.embed_query(question))
        scores = []
        for chunk in chunks:
            chunk_emb = np.array(embedding_model.embed_query(chunk))
            distance = np.linalg.norm(question_emb - chunk_emb)
            # close to 1 -> Good matching, close to 0 -> Bad matching
            scores.append((1 / (1 + distance)))
            
        return scores

    def rerank(self, state: AgentState) -> AgentState:
        """
        Calculate the weighted average chunk's score 

        Parameters:
            state (AgentState): the graph state

        Returns:
            AgentState: the updated graph state
        
        Raises:
            NotImplementedError: if the reranking strategy is not supported
            ValueError: if the reranking weights do not sum up to 1
        """
        question = state["original_question"]
        scores_per_strategy = []
        chunks = state["chunks"]
        # Calculate score for each strategy
        for strategy in self._strategies:
            if strategy == "semantic":
                scores_per_strategy.append(self._calculate_semantic_score(question, chunks))
            elif strategy == "distance":
                scores_per_strategy.append(self._calculate_distance_score(question, chunks))
            else:
                raise NotImplementedError(f"reranking strategy {strategy} not supported")

        # Numpy arrays for fast computation and built-in methods
        matrix = np.array(scores_per_strategy)
        weights = np.array(self._weights)

        if weights.sum() != 1:
            raise ValueError("reranking weights must sum to 1")

        # Weighted Average on all strategies
        final_score = np.average(matrix, axis=0, weights=weights).tolist()
        
        state["messages"].append(AIMessage(f"weighted average score between all reranking techniques: {final_score}"))
        state["reranking_score"] = final_score

        return state