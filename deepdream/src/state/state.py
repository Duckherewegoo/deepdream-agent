from typing import Annotated, Any, TypedDict

from langchain_core.messages.base import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_input: str
    image_path: str | None
    output_image_path: str | None
    dream_type: str | None
    need_dream_gen: bool
    tool_results: list[dict[str, Any]]
    final_output: str | None
    loop_count: int
    status: str
    error: str | None
