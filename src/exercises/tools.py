import logging
from langchain_core.tools import tool
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode

import tools as my_custom_tools

logger = logging.getLogger(__name__)

@tool
def cancel_order(order_id: str) -> str:
    """Cancel an order that hasn't shipped"""
    logger.info("==========cancelling....============")
    return f"DEBUG: Order {order_id} has been cancelled."


@tool
def check_shipping(order_id: str) -> str:
    """Check the real-time shipping tracking coordinates for an order."""
    logger.info(f"========== AUTOMATIC EXECUTION: checking shipping for {order_id} ============")
    return f"DEBUG: Order {order_id} is currently in transit."


tools_list = [
    getattr(my_custom_tools, attribute_name)
    for attribute_name in dir(my_custom_tools)
    if isinstance(getattr(my_custom_tools, attribute_name), BaseTool)
]

tool_node = ToolNode(tools_list)
tool_lookup_map = {tool_obj.name: tool_obj for tool_obj in tools_list}