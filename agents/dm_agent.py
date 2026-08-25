from typing import Any, Dict, List, Union
from .llm_gpt import LLM_gpt

class DM_agent:
    """
    Decision-Making Agent for BDI household tasks.
    Wraps LLM to generate Belief-Desire-Intention thoughts and Actions.
    """
    def __init__(self, llm: Any = None, **kwargs: Any):
        llm_class = kwargs.pop("llm_class", LLM_gpt)
        self.llm = llm if llm is not None else llm_class(**kwargs)

    def generate(self, messages: Union[str, List[Dict[str, Any]]]) -> str:
        """ Generate action and thought from decision agent. """
        return self.llm.generate(messages)

    def generate_action(self, messages: Union[str, List[Dict[str, Any]]]) -> str:
        """ Generate action and thought from decision agent. """
        return self.llm.generate(messages)
