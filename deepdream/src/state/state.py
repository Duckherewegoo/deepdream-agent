from typing import Any, TypedDict

from langchain_core.messages.base import BaseMessage


class AgentState(TypedDict):
    messages: list[BaseMessage]
    user_input: str
    image_path: str | None
    output_image_path: str | None
    dream_type: str | None
    need_dream_gen: bool
    tool_results: list[dict[str, Any]]
    final_output: str | None
    status: str
    error: str | None
