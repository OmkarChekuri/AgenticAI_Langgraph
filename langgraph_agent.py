import base64
import re
from typing import TypedDict, List, Literal, Union
import httpx
import streamlit as st # Streamlit is imported here for st.error/st.warning in helper functions

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

# ---- STATE DEFINITION ----
class GraphState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    model_choice: Literal["vision", "text", "code", "auto"]
    next_node_hint: Literal["vision_handler", "text_handler", "code_handler", "llm_router_node"]


# ---- MODEL INITIALIZATION ----
@st.cache_resource
def load_llm_models():
    """Loads and caches Ollama LLM models."""
    return {
        "llm_text": ChatOllama(model="llama3.2"),
        "llm_vision": ChatOllama(model="gemma3"),
        "llm_code": ChatOllama(model="qwen2.5-coder:7b")
    }

llm_models = load_llm_models()
llm_text = llm_models["llm_text"]
llm_vision = llm_models["llm_vision"]
llm_code = llm_models["llm_code"]


# ---- HELPER FUNCTIONS (for image processing) ----
def encode_image_to_base64(image_bytes: bytes) -> str:
    """Encodes image bytes to a base64 string."""
    return base64.b64encode(image_bytes).decode('utf-8')

def fetch_and_encode_web_image(image_url: str) -> Union[str, None]:
    """Fetches an image from a URL and encodes it to a base64 string."""
    #print(f"Fetching and encoding web image from URL: {image_url}") # For console debugging
    try:
        response = httpx.get(image_url, follow_redirects=True, timeout=10)
        response.raise_for_status()
        return base64.b64encode(response.content).decode("utf-8")
    except httpx.RequestError as e:
        st.error(f"Error fetching image from URL {image_url}: {e}")
        return None
    except Exception as e:
        st.error(f"Error encoding image from URL {image_url}: {e}")
        return None


# ---- GRAPH NODES ----
def text_node(state: GraphState) -> GraphState:
    """Processes the message using the text-capable LLM."""
    print("---Entering TEXT NODE---")
    response = llm_text.invoke(state["messages"])
    # Store the hint that led to this node in additional_kwargs
    state["messages"].append(AIMessage(content=response.content, additional_kwargs={"llm_hint": "text_handler"}))
    return state

def vision_node(state: GraphState) -> GraphState:
    """Processes the message using the vision-capable LLM."""
    print("---Entering VISION NODE---")
    #print(f"Vision Node - Incoming Messages: {state['messages']}") # Debugging line
    try:
        response = llm_vision.invoke(state["messages"])
        if response and response.content:
            state["messages"].append(AIMessage(content=response.content, additional_kwargs={"llm_hint": "vision_handler"}))
        else:
            state["messages"].append(AIMessage(content="Vision model did not provide a clear response. It might be struggling to interpret the image or the query.", additional_kwargs={"llm_hint": "vision_handler_fallback"}))
            print("Warning: Vision model returned empty or no content.")
    except Exception as e:
        error_message = f"An error occurred while processing the image with the vision model: {e}. Please ensure the model is running and compatible with the image format."
        state["messages"].append(AIMessage(content=error_message, additional_kwargs={"llm_hint": "vision_handler_error"}))
        st.error(error_message)
        print(f"Error in vision_node: {e}")
    return state

def code_node(state: GraphState) -> GraphState:
    """Processes the message using the code-capable LLM."""
    print("---Entering CODE NODE---")
    response = llm_code.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content, additional_kwargs={"llm_hint": "code_handler"}))
    return state

def llm_router_node(state: GraphState) -> GraphState:
    """Uses the text LLM to decide the next routing path ('text', 'vision', or 'code')."""
    print("---Entering LLM ROUTER NODE---")
    last_user_message_content = ""
    last_msg = state["messages"][-1]
    if isinstance(last_msg, HumanMessage):
        if isinstance(last_msg.content, str):
            last_user_message_content = last_msg.content
        elif isinstance(last_msg.content, list):
            for part in last_msg.content:
                if isinstance(part, dict) and part.get("type") == "text":
                    last_user_message_content = part.get("text", "")
                    break

    if not last_user_message_content:
        print("No relevant text content found for LLM routing. Defaulting to text_handler.")
        state["next_node_hint"] = "text_handler"
        return state

    routing_prompt = [
        HumanMessage(content=f"You are an intelligent router. Based on the following user query, decide if it primarily requires a 'text' model, a 'vision' model, or a 'code' model. Respond with ONLY one word: 'text', 'vision', or 'code'.\n\nUser query: {last_user_message_content}")
    ]

    try:
        response = llm_text.invoke(routing_prompt)
        decision = response.content.strip().lower()
    except Exception as e:
        st.error(f"Error invoking LLM for routing: {e}. Defaulting to text_handler.")
        decision = "text"

    if "vision" in decision:
        state["next_node_hint"] = "vision_handler"
    elif "code" in decision:
        state["next_node_hint"] = "code_handler"
    else:
        state["next_node_hint"] = "text_handler"

    print(f"LLM Router decided next hint: {state['next_node_hint']}")
    if 'current_llm_hint' in st.session_state: # Only update if in Streamlit context
        st.session_state.current_llm_hint = state['next_node_hint']
    return state

def preprocess_and_route_node(state: GraphState) -> GraphState:
    """
    Preprocesses the last message (e.g., checks for multimodal content) and
    sets the hint for the next routing decision.
    """
    print("---Entering PREPROCESS AND ROUTE NODE---")
    last_msg = state["messages"][-1]
    
    if "model_choice" not in state:
        state["model_choice"] = "auto"

    next_hint_from_preprocess = None
    
    # Check if the message is already multimodal (meaning image was processed by UI logic)
    if isinstance(last_msg, HumanMessage) and isinstance(last_msg.content, list):
        for part in last_msg.content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                next_hint_from_preprocess = "vision_handler"
                print("Multimodal message with image detected (from UI preprocessing), directly routing to VISION HANDLER.")
                break

    # If no direct routing via image detection, proceed with explicit model_choice or LLM routing
    if next_hint_from_preprocess is None:
        if state.get("model_choice") == "code":
            next_hint_from_preprocess = "code_handler"
            state["model_choice"] = "auto"
        elif state.get("model_choice") == "vision":
            next_hint_from_preprocess = "vision_handler"
            state["model_choice"] = "auto"
        else:
            next_hint_from_preprocess = "llm_router_node"
            if isinstance(last_msg, HumanMessage) and isinstance(last_msg.content, str):
                code_keywords = ["code", "program", "function", "script", "develop", "implement", "write in",
                                 "python", "javascript", "java", "c++", "html", "css", "sql", "bash"]
                if any(keyword in last_msg.content.lower() for keyword in code_keywords):
                    print("Code keywords detected, routing to LLM Router for decision.")


    state["next_node_hint"] = next_hint_from_preprocess
    print(f"Preprocess node decided next hint: {next_hint_from_preprocess}")
    if 'current_llm_hint' in st.session_state: # Only update if in Streamlit context
        st.session_state.current_llm_hint = state['next_node_hint']
    return state


# ---- ROUTING DECISION FUNCTION ----
def route_decision(state: GraphState) -> Literal["vision_handler", "text_handler", "code_handler", "llm_router_node"]:
    """Reads the 'next_node_hint' from the state to determine the next node."""
    print(f"---Making ROUTING DECISION based on hint: {state['next_node_hint']}---")
    return state["next_node_hint"]


# ---- LANGGRAPH WORKFLOW COMPILATION ----
@st.cache_resource
def compile_langgraph_workflow():
    """Compiles the Langgraph workflow."""
    workflow = StateGraph(GraphState)

    workflow.add_node("preprocess_and_route_node", preprocess_and_route_node)
    workflow.add_node("llm_router_node", llm_router_node)
    workflow.add_node("text_handler", text_node)
    workflow.add_node("vision_handler", vision_node)
    workflow.add_node("code_handler", code_node)

    workflow.set_entry_point("preprocess_and_route_node")

    workflow.add_conditional_edges(
        "preprocess_and_route_node",
        route_decision,
        {
            "text_handler": "text_handler",
            "vision_handler": "vision_handler",
            "code_handler": "code_handler",
            "llm_router_node": "llm_router_node"
        }
    )

    workflow.add_conditional_edges(
        "llm_router_node",
        route_decision,
        {
            "text_handler": "text_handler",
            "vision_handler": "vision_handler",
            "code_handler": "code_handler"
        }
    )

    workflow.add_edge("text_handler", END)
    workflow.add_edge("vision_handler", END)
    workflow.add_edge("code_handler", END)

    return workflow.compile()

