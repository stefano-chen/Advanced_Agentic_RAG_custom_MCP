from typing import List, Dict, Any, Tuple
from utils.state import AgentState
from langchain_core.messages import AIMessage

class ChunckSelection:
    """
    The Selection Node
    """
    def __init__(self, strategies: List[str], options: Dict[str, Dict[str, Any]]):
        """
        Attributes:
            stategies (List[str]): a list of string representing the selection strategies to apply (e.g. ["threshold", "topk"])
            options: (Dict[str, Dict[str, Any]]): a dictionary that for each stratety, contains the respective parameters
        """
        self._strategies = strategies
        self._options = options

    def _selection_by_threshold(self, chunks: List[str], scores: List[float], options: Dict[str, Any]) -> Tuple[List[str], List[float]]:
        """
        Given a list of chunks and the respective scores, this function selects only the chuncks with a score higher greater or equal to a threshold

        Parameters:
            chunks (List[str]): a list of chunks
            scores (List[float]): the score for each chunk
            options (Dict[str, Any]): contains the options for this strategy
        
        Returns:
            A tuple containing, respectively, a list of selected chunks and a list of scores
        """
        if not options:
            raise Exception("threshold options not found")
        if "min" not in options:
            raise Exception("min field not found in threshold options")
        threshold = options["min"]
        selected_chunks = []
        selected_scores = []
        for i in range(len(chunks)):
            if scores[i] >= threshold:
                selected_chunks.append(chunks[i])
                selected_scores.append(scores[i])
        return selected_chunks, selected_scores

    def _selection_by_topk(self, chunks: List[str], scores: List[float], options: Dict[str, Any]) -> Tuple[List[str], List[float]]:
        """
        Given a list of chunks and the respective scores, this function selects the top k chuncks with the highest scores

        Parameters:
            chunks (List[str]): a list of chunks
            scores (List[float]): the score for each chunk
            options (Dict[str, Any]): contains the options for this strategy
        
        Returns:
            A tuple containing, respectively, a list of selected chunks and a list of scores
        """
        if not options:
            raise Exception("topk options not found")
        if "k" not in options:
            raise Exception("k field not found in topk options")
        k = options["k"]
        # Get sorted indices based on scores
        indices = sorted(range(len(scores)), key=lambda i : scores[i], reverse=True)
        # Reorder chunks using sorted indices
        sorted_chunks = [chunks[i] for i in indices]
        sorted_scores = [scores[i] for i in indices]

        return sorted_chunks[:k], sorted_scores[:k] 

    def select(self, state: AgentState) -> AgentState:
        """
        Use the selection techniques to select chunks

        Parameters:
            state (AgentState): the graph state

        Returns:
            AgentState: the updated graph state
        
        Raises:
            NotImplementedError: if the selection strategy is not supported
        """
        chunks = state['chunks']
        scores = state['reranking_score']

        for strategy in self._strategies:
            if strategy == "threshold":
                strategy_option = self._options.get("threshold")
                chunks, scores = self._selection_by_threshold(chunks, scores, strategy_option)
            elif strategy == "topk":
                strategy_option = self._options.get("topk")
                chunks, scores = self._selection_by_topk(chunks, scores, strategy_option)
            else:
                raise NotImplementedError(f"selection strategy {strategy} not supported")
        
        state["messages"].append(AIMessage(f"{len(chunks)} chunks selected"))
        state["chunks"] = chunks
        state["reranking_score"] = None

        return state


