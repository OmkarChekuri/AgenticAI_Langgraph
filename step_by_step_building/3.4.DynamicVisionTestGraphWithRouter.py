import base64
import re # For regex to detect file paths
from typing import TypedDict, List, Literal, Union
import httpx # New import for fetching web images

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

# ---- STATE DEFINITION ----
# Defines the structure of the graph's state.
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    Attributes:
        messages: A list of messages exchanged in the conversation.
                  Can contain both HumanMessage (user input) and AIMessage (agent output).
        model_choice: An optional hint for the router to explicitly select a model for the first turn.
                      This is reset after the first use to allow content-based routing.
        next_node_hint: A hint set by the preprocess_and_route_node to guide the next transition.
    """
    messages: List[Union[HumanMessage, AIMessage]]
    model_choice: Literal["vision", "text", "code", "auto"] # 'auto' for default content-based routing
    next_node_hint: Literal["vision_handler", "text_handler", "code_handler"] # Now includes vision and code


# ---- MODEL INITIALIZATION ----
# Initialize different Ollama models for specific tasks.
# Ensure these models are downloaded and available via your Ollama server.
# You can download them using: ollama pull <model_name> (e.g., ollama pull llava)
llm_text = ChatOllama(model="llama3.2")         # General text generation
llm_vision = ChatOllama(model="gemma3")          # Vision-capable model (recommended for image understanding)
llm_code = ChatOllama(model="qwen2.5-coder:7b")    # Code generation/understanding


# ---- HELPER FUNCTION: FETCH AND ENCODE WEB IMAGE ----
def fetch_and_encode_web_image(image_url: str) -> Union[str, None]:
    """
    Fetches an image from a URL and encodes it to a base64 string.

    Args:
        image_url: The URL of the image.

    Returns:
        The base64 encoded string of the image, or None if an error occurs.
    """
    try:
        response = httpx.get(image_url, follow_redirects=True, timeout=10)
        response.raise_for_status() # Raise an exception for bad status codes
        return base64.b64encode(response.content).decode("utf-8")
    except httpx.RequestError as e:
        print(f"Error fetching image from URL {image_url}: {e}")
        return None
    except Exception as e:
        print(f"Error encoding image from URL {image_url}: {e}")
        return None


# ---- GRAPH NODES ----
def text_node(state: GraphState) -> GraphState:
    """
    Processes the message using the text-capable LLM.
    Appends the AI's response to the messages list.
    """
    print("---Entering TEXT NODE---")
    response = llm_text.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    return state

def vision_node(state: GraphState) -> GraphState:
    """
    Processes the message using the vision-capable LLM.
    This node expects the message to already contain image data,
    which is handled by the preprocess_and_route_node.
    Appends the AI's response to the messages list.
    """
    print("---Entering VISION NODE---")
    response = llm_vision.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    return state

def code_node(state: GraphState) -> GraphState:
    """
    Processes the message using the code-capable LLM.
    Appends the AI's response to the messages list.
    """
    print("---Entering CODE NODE---")
    response = llm_code.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    return state


# ---- PREPROCESSING AND ROUTING NODE (Web Image Support) ----
def preprocess_and_route_node(state: GraphState) -> GraphState:
    """
    Preprocesses the last message (e.g., fetches and encodes web images) and sets a hint
    in the state for the next routing decision based on content.
    """
    print("---Entering PREPROCESS AND ROUTE NODE---")
    last_msg = state["messages"][-1]
    next_hint = "text_handler" # Default hint for next node

    # 1. Check for explicit 'model_choice' hint (useful for initial testing or forced routing)
    if state.get("model_choice") == "code":
        next_hint = "code_handler"
        state["model_choice"] = "auto" # Reset to auto for next turn
    elif state.get("model_choice") == "vision":
        next_hint = "vision_handler"
        state["model_choice"] = "auto" # Reset to auto for next turn
    
    # If a hint was already set by model_choice, we prioritize it.
    # Otherwise, proceed with content-based routing.
    if next_hint == "text_handler": # Only proceed if no explicit hint was given
        if isinstance(last_msg, HumanMessage) and last_msg.content:
            # 2. Check for multimodal image content (if already pre-formatted)
            if isinstance(last_msg.content, list): # Check if content is already a list (multimodal)
                for part in last_msg.content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        next_hint = "vision_handler"
                        break # Found image, no need to check further
            elif isinstance(last_msg.content, str): # Check if content is a string (text or URL)
                text_content = last_msg.content.lower()

                # 3. Check for web image URLs in the text content
                # This regex looks for common http/https URLs pointing to image files.
                web_image_url_match = re.search(r'(https?:\/\/[^\s\/$.?#].[^\s]*?\.(jpg|jpeg|png|gif|bmp|tiff))', text_content)
                if web_image_url_match:
                    detected_url = web_image_url_match.group(1) # Extract the full URL
                    image_extension = web_image_url_match.group(2) # Extract the extension for mime type
                    print(f"Detected potential web image URL: {detected_url}")
                    
                    base64_img = fetch_and_encode_web_image(detected_url) # Fetch and encode the web image
                    if base64_img:
                        # If image successfully encoded, transform the HumanMessage into a multimodal format
                        new_text_part = text_content.replace(web_image_url_match.group(0), "").strip() or "Describe this image:"
                        new_content = [
                            {"type": "text", "text": new_text_part},
                            {"type": "image_url", "image_url": {"url": f"data:image/{image_extension};base64,{base64_img}"}}, # Use data: URI
                        ]
                        state["messages"][-1] = HumanMessage(content=new_content)
                        next_hint = "vision_handler"
                        print(f"Routing to VISION HANDLER with base64 encoded web image.")
                    else:
                        print("Failed to fetch or encode web image, defaulting to TEXT HANDLER.")
                        next_hint = "text_handler" # Fallback if image fetching/encoding fails

                # 4. Check for code-related keywords in the text content
                code_keywords = ["code", "program", "function", "script", "develop", "implement", "write in",
                                 "python", "javascript", "java", "c++", "html", "css", "sql", "bash"]
                if any(keyword in text_content for keyword in code_keywords):
                    next_hint = "code_handler"

    state["next_node_hint"] = next_hint # Set the hint in the state
    print(f"Preprocess node decided next hint: {next_hint}")
    return state


# ---- ROUTING DECISION FUNCTION ----
def route_decision(state: GraphState) -> Literal["vision_handler", "text_handler", "code_handler"]:
    """
    Reads the 'next_node_hint' from the state to determine the next node.
    This function is used by add_conditional_edges and only returns a string.
    """
    print(f"---Making ROUTING DECISION based on hint: {state['next_node_hint']}---")
    return state["next_node_hint"]


# ---- BUILD THE LANGGRAPH WORKFLOW ----
workflow = StateGraph(GraphState)

# Add all nodes
workflow.add_node("preprocess_and_route_node", preprocess_and_route_node)
workflow.add_node("text_handler", text_node)
workflow.add_node("vision_handler", vision_node)
workflow.add_node("code_handler", code_node)

# Set the preprocess_and_route_node as the entry point
workflow.set_entry_point("preprocess_and_route_node")

# Add conditional edges from the preprocess_and_route_node
workflow.add_conditional_edges(
    "preprocess_and_route_node", # Source node
    route_decision,              # Function to determine next step
    {
        "text_handler": "text_handler",
        "vision_handler": "vision_handler",
        "code_handler": "code_handler"
    }
)

# All handlers now go to END for a single turn
workflow.add_edge("text_handler", END)
workflow.add_edge("vision_handler", END)
workflow.add_edge("code_handler", END)

# Compile the graph
graph = workflow.compile()


# ---- TEST CASES ----
def run_single_turn_test(name: str, input_messages: List[HumanMessage], initial_model_choice: Literal["vision", "text", "code", "auto"] = "auto"):
    """
    Runs a single turn test case for the Langgraph workflow.
    """
    print(f"\n--- Running Single Turn Test: {name} ---")
    input_state = {
        "messages": input_messages,
        "model_choice": initial_model_choice,
        "next_node_hint": "text_handler" # Default, will be overridden by preprocess_and_route_node
    }
    
    result = graph.invoke(input_state)
    
    print(f"Agent Response: {result['messages'][-1].content}")
    print(f"Total messages in conversation history: {len(result['messages'])}")



# --- Test 5: Image Description Request (explicit hint, another web URL) ---
run_single_turn_test(
    "Image Description Request (explicit hint, another web URL)",
    [HumanMessage(content="Describe this picture in two lines: https://homedecorez.co.uk/wp-content/uploads/2024/12/Practical-Magic-House.webp")],
    initial_model_choice="vision"
)


# --- Test 1: Simple Text Query ---
run_single_turn_test(
    "Simple Text Query",
    [HumanMessage(content="What is the capital of Canada?")],
    initial_model_choice="auto"
)

# --- Test 2: Code Generation Request (keyword-based) ---
run_single_turn_test(
    "Code Generation Request (keywords)",
    [HumanMessage(content="Write a Python function to reverse a string.")],
    initial_model_choice="auto"
)

# --- Test 3: Image Description Request (using a web URL) ---
# This uses the example URL you provided.
run_single_turn_test(
    "Image Description Request (web URL)",
    [HumanMessage(content="Describe the weather in this image in two lines: file:///D:/ML/AgenticAI_Projects/AgenticAI_langgraph/R.jpeg")],
    initial_model_choice="auto"
)

# --- Test 4: Code Generation Request (explicit hint) ---
run_single_turn_test(
    "Code Generation Request (explicit hint)",
    [HumanMessage(content="How do I implement a quicksort algorithm in Java?")],
    initial_model_choice="code"
)
