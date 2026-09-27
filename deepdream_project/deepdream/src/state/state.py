from typing import Any, Dict, List, TypedDict, Optional
from langchain_core.messages.base import BaseMessage


class AgentState(TypedDict):
    messages: List[BaseMessage]
    user_input: str
    image_path: Optional[str]
    output_image_path: Optional[str]
    dream_type: Optional[str]
    need_dream_gen: bool
    tool_results: List[Dict[str, Any]]
    final_output: Optional[str]
    status: str
    error: Optional[str]
