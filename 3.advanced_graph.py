from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, List, Literal, Union

# ---- STATE ----
class GraphState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    model_choice: Literal["vision", "text"]

# ---- MODELS ----
llm_text = ChatOllama(model="llama3.2")  # Text model
llm_vision = ChatOllama(model="gemma3")         # Vision-capable model (or replace with llava)

# ---- TEXT NODE ----
def text_node(state: GraphState) -> GraphState:
    response = llm_text.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    return state

# ---- VISION NODE ----
def vision_node(state: GraphState) -> GraphState:
    response = llm_vision.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    return state

# ---- ROUTER ----
def route(state: GraphState) -> Literal["vision_handler", "text_handler"]:
    last_msg = state["messages"][-1]
    if isinstance(last_msg, HumanMessage) and "file://" in last_msg.content:
        return "vision_handler"
    return "text_handler"

# ---- BUILD GRAPH ----
workflow = StateGraph(GraphState)
workflow.add_node("text_handler", text_node)
workflow.add_node("vision_handler", vision_node)
workflow.set_conditional_entry_point(
    route,
    {
        "text_handler": "text_handler",
        "vision_handler": "vision_handler"
    }
)
workflow.add_edge("text_handler", END)
workflow.add_edge("vision_handler", END)
graph = workflow.compile()

# ---- TEST TEXT INPUT ----
text_input = {
    "messages": [HumanMessage(content="What's the capital of Brazil?")],
    "model_choice": "text"
}

print("=== TEXT INPUT TEST ===")
result = graph.invoke(text_input)
print(result["messages"][-1].content)

# ---- TEST IMAGE INPUT ----
image_input = {
    "messages": [
        HumanMessage(content=f"Describe this image: D:\ML\AgenticAI_Projects\AgenticAI_langgraph\R.jpeg")
    ],
    "model_choice": "vision"
}

print("\n=== IMAGE INPUT TEST ===")
result = graph.invoke(image_input)
print(result["messages"][-1].content)
