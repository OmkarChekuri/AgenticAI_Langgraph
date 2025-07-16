import streamlit as st
import base64
import re # For regex to detect file paths
from typing import TypedDict, List, Literal, Union
import httpx # For fetching web images
import io # For handling uploaded file bytes (and graph image bytes)

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

# --- 1. LANGGRAPH WORKFLOW DEFINITION ---

# ---- STATE DEFINITION ----
class GraphState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    model_choice: Literal["vision", "text", "code", "auto"]
    next_node_hint: Literal["vision_handler", "text_handler", "code_handler", "llm_router_node"]


# ---- MODEL INITIALIZATION ----
# Use st.cache_resource to cache LLM models, preventing re-initialization on every rerun
@st.cache_resource
def load_llm_models():
    return {
        "llm_text": ChatOllama(model="llama3.2"),
        "llm_vision": ChatOllama(model="gemma3"), # Reverted to gemma3 as per your request
        "llm_code": ChatOllama(model="qwen2.5-coder:7b")
    }

llm_models = load_llm_models()
llm_text = llm_models["llm_text"]
llm_vision = llm_models["llm_vision"]
llm_code = llm_models["llm_code"]


# ---- HELPER FUNCTIONS ----
def encode_local_image_bytes(image_bytes: bytes) -> str:
    """
    Encodes image bytes (from an uploaded file) to a base64 string.
    """
    return base64.b64encode(image_bytes).decode('utf-8')

def fetch_and_encode_web_image(image_url: str) -> Union[str, None]:
    """
    Fetches an image from a URL and encodes it to a base64 string.
    """
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
    print("---Entering TEXT NODE---")
    response = llm_text.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    return state

def vision_node(state: GraphState) -> GraphState:
    print("---Entering VISION NODE---")
    # st.write(f"Vision Node - Incoming Messages: {state['messages']}") # Debugging line, uncomment if needed
    response = llm_vision.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    return state

def code_node(state: GraphState) -> GraphState:
    print("---Entering CODE NODE---")
    response = llm_code.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    return state


# ---- LLM-DRIVEN ROUTER NODE ----
def llm_router_node(state: GraphState) -> GraphState:
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
    return state


# ---- PREPROCESSING AND ROUTING NODE (Handles Uploaded, Local, and Web Images) ----
def preprocess_and_route_node(state: GraphState) -> GraphState:
    print("---Entering PREPROCESS AND ROUTE NODE---")
    last_msg = state["messages"][-1]
    
    if "model_choice" not in state:
        state["model_choice"] = "auto"

    next_hint_from_preprocess = None
    image_processed = False

    # --- NEW: Handle uploaded image from Streamlit session state ---
    if st.session_state.get("uploaded_image_data") and st.session_state.get("uploaded_image_mime_type"):
        # Create a multimodal message from the uploaded image
        # Use existing text content if available, otherwise a default prompt
        text_part = last_msg.content if isinstance(last_msg.content, str) else "Describe this image:"
        new_content = [
            {"type": "text", "text": text_part},
            {"type": "image_url", "image_url": {"url": f"data:{st.session_state.uploaded_image_mime_type};base64,{st.session_state.uploaded_image_data}"}},
        ]
        state["messages"][-1] = HumanMessage(content=new_content)
        image_processed = True
        print("Uploaded image processed and message updated.")
        # Clear uploaded image data from session state after processing
        st.session_state.uploaded_image_data = None
        st.session_state.uploaded_image_mime_type = None

    # If no uploaded image, proceed with text-based image detection (local/web)
    if not image_processed and isinstance(last_msg, HumanMessage) and isinstance(last_msg.content, str):
        text_content = last_msg.content

        # 1. Check for LOCAL image file paths
        local_file_path_match = re.search(
            r'(file://)?'                                 
            r'('                                           
            r'(?:[a-zA-Z]:[\\/]|[\/])'                     
            r'(?:[^<>:"|?*\\/]+\\?)*'                  
            r'[^<>:"|?*\\/]*'                             
            r'\.'                                         
            r'(jpg|jpeg|png|gif|bmp|tiff)'                
            r')'                                           
            r'$',                                          
            text_content,
            re.IGNORECASE
        )

        if local_file_path_match:
            detected_path = local_file_path_match.group(2)
            image_extension = local_file_path_match.group(3)
            print(f"Detected potential local image file path: {detected_path}")
            
            try: # Added try-except for file reading
                base64_img = encode_local_image_bytes(open(detected_path, "rb").read()) # Read local file bytes
                if base64_img:
                    new_text_part = text_content.replace(local_file_path_match.group(0), "").strip() or "Describe this image:"
                    new_content = [
                        {"type": "text", "text": new_text_part},
                        {"type": "image_url", "image_url": {"url": f"data:image/{image_extension};base64,{base64_img}"}},
                    ]
                    state["messages"][-1] = HumanMessage(content=new_content)
                    image_processed = True
                    print(f"Local image encoded and message updated.")
                else:
                    st.warning("Failed to encode local image from path.")
            except FileNotFoundError:
                st.warning(f"Local image file not found: {detected_path}")
            except Exception as e:
                st.warning(f"Error processing local image: {e}")


        # 2. Check for WEB image URLs (only if no local image was found and processed)
        if not image_processed:
            web_image_url_match = re.search(r'(https?:\/\/[^\s\/$.?#].[^\s]*?\.(jpg|jpeg|png|gif|bmp|tiff))', text_content, re.IGNORECASE)
            if web_image_url_match:
                detected_url = web_image_url_match.group(1)
                image_extension = web_image_url_match.group(2)
                print(f"Detected potential web image URL: {detected_url}")
                
                base64_img = fetch_and_encode_web_image(detected_url)
                if base64_img:
                    new_text_part = text_content.replace(web_image_url_match.group(0), "").strip() or "Describe this image:"
                    new_content = [
                        {"type": "text", "text": new_text_part},
                        {"type": "image_url", "image_url": {"url": f"data:image/{image_extension};base64,{base64_img}"}},
                    ]
                    state["messages"][-1] = HumanMessage(content=new_content)
                    image_processed = True
                    print(f"Web image encoded and message updated.")
                else:
                    st.warning("Failed to fetch or encode web image.")
    
    # Now that image preprocessing is done, determine the next hint based on model_choice or LLM routing
    if state.get("model_choice") == "code":
        next_hint_from_preprocess = "code_handler"
        state["model_choice"] = "auto"
    elif state.get("model_choice") == "vision":
        next_hint_from_preprocess = "vision_handler"
        state["model_choice"] = "auto"
    elif image_processed: # If an image was processed (uploaded, local, or web), directly route to vision_handler
        next_hint_from_preprocess = "vision_handler"
        print("Image successfully processed, directly routing to VISION HANDLER.")
    else:
        # If no explicit model_choice AND no image was processed,
        # then delegate to the LLM router for text/code keyword classification.
        next_hint_from_preprocess = "llm_router_node"
        # Optional: Check for code keywords here to influence LLM router's decision if needed.
        # The LLM router's prompt is designed to handle this from text content.


    state["next_node_hint"] = next_hint_from_preprocess
    print(f"Preprocess node decided next hint: {next_hint_from_preprocess}")
    return state


# ---- ROUTING DECISION FUNCTION ----
def route_decision(state: GraphState) -> Literal["vision_handler", "text_handler", "code_handler", "llm_router_node"]:
    print(f"---Making ROUTING DECISION based on hint: {state['next_node_hint']}---")
    return state["next_node_hint"]


# ---- BUILD THE LANGGRAPH WORKFLOW ----
@st.cache_resource
def compile_langgraph_workflow():
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

graph = compile_langgraph_workflow()

# --- 2. STREAMLIT APPLICATION ---

st.set_page_config(page_title="Dynamic Langgraph Agent", layout="centered")

st.title("🤖 Dynamic Langgraph Agent")
st.markdown("""
This agent can dynamically route your queries to different LLMs (Text, Vision, Code)
based on your input. You can type questions, ask for code, or provide image URLs/upload images!
""")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_state" not in st.session_state:
    st.session_state.conversation_state = {
        "messages": [],
        "model_choice": "auto",
        "next_node_hint": "text_handler"
    }
if "uploaded_image_data" not in st.session_state:
    st.session_state.uploaded_image_data = None
if "uploaded_image_mime_type" not in st.session_state:
    st.session_state.uploaded_image_mime_type = None
# Initialize new session state for displaying current LLM hint
if "current_llm_hint" not in st.session_state:
    st.session_state.current_llm_hint = "Waiting for input..."

# --- Display Chat Messages ---
chat_container = st.container(height=400, border=True)
with chat_container:
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                if isinstance(message.content, str):
                    st.markdown(message.content)
                elif isinstance(message.content, list):
                    # Handle multimodal content for user messages
                    for part in message.content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                st.markdown(part.get("text"))
                            elif part.get("type") == "image_url":
                                image_url_data = part.get("image_url", {}).get("url")
                                if image_url_data and image_url_data.startswith("data:image"):
                                    st.image(image_url_data, caption="Uploaded Image", width=200)
                                else:
                                    st.markdown(f"Image URL: {image_url_data}") # Fallback for non-data URLs
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

# --- User Input Area ---
user_query = st.chat_input("Type your message here...")

# --- Image Upload ---
uploaded_file = st.file_uploader("Or upload an image:", type=["png", "jpg", "jpeg", "gif", "bmp", "tiff"])

# Display the current LLM hint
st.info(f"Current LLM: {st.session_state.current_llm_hint.replace('_handler', '').replace('_node', '').title()}")


if uploaded_file is not None:
    # Read image bytes and store in session state
    image_bytes = uploaded_file.read()
    st.session_state.uploaded_image_data = encode_local_image_bytes(image_bytes)
    st.session_state.uploaded_image_mime_type = uploaded_file.type
    
    # Add a placeholder message to trigger the graph processing
    # The actual image data is in session_state, preprocess_and_route_node will pick it up
    st.session_state.messages.append(HumanMessage(content=f"Image uploaded: {uploaded_file.name}"))
    user_query = "Describe the uploaded image." # Use a default text query to accompany the image

# --- Process User Input ---
if user_query:
    # Append user's message to display history (if not already added by file uploader)
    if not st.session_state.messages or st.session_state.messages[-1].content != user_query:
        # If it's a new text query or not an image upload, add it to history
        if not (uploaded_file is not None and user_query == "Describe the uploaded image."):
             st.session_state.messages.append(HumanMessage(content=user_query))

    # Update the Langgraph conversation state with the new user message
    # If an image was uploaded, the HumanMessage content will be updated by preprocess_and_route_node
    # If not, it will be the plain text query.
    st.session_state.conversation_state["messages"].append(HumanMessage(content=user_query))

    with st.spinner("Thinking..."):
        try:
            # Invoke the graph for one turn
            st.session_state.conversation_state = graph.invoke(st.session_state.conversation_state)
            
            # Update the current LLM hint after invocation
            st.session_state.current_llm_hint = st.session_state.conversation_state["next_node_hint"]

            # Get the agent's response from the updated conversation state
            agent_response_msg = st.session_state.conversation_state["messages"][-1]
            st.session_state.messages.append(agent_response_msg) # Add agent's response to display history
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.warning("Please try again.")
            # Optionally, remove the last user message from Langgraph state to avoid re-processing
            if st.session_state.conversation_state["messages"] and isinstance(st.session_state.conversation_state["messages"][-1], HumanMessage):
                conversation_state["messages"].pop() # Fix: use conversation_state, not st.session_state.conversation_state
                
    st.rerun() # Rerun to update chat display


# --- Clear Chat Button ---
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.conversation_state = {
        "messages": [],
        "model_choice": "auto",
        "next_node_hint": "text_handler"
    }
    st.session_state.uploaded_image_data = None
    st.session_state.uploaded_image_mime_type = None
    st.session_state.current_llm_hint = "Waiting for input..." # Reset hint
    st.rerun()

# --- Workflow Graph Visualization (in sidebar) ---
st.sidebar.header("Workflow Graph")
with st.sidebar.expander("Show Graph"):
    try:
        # Draw the graph to an in-memory BytesIO object
        graph_bytes_io = io.BytesIO()
        graph.get_graph().draw_png(graph_bytes_io, prog="dot") # Use 'dot' program for layout
        graph_bytes_io.seek(0) # Rewind to the beginning of the BytesIO object

        st.image(graph_bytes_io, caption="Langgraph Workflow", use_column_width=True)
    except Exception as e:
        st.warning(f"Could not draw graph: {e}")
        st.info("Please ensure 'pygraphviz' and 'graphviz' are installed correctly on your system.")
        st.markdown("For `graphviz`, you might need a system-wide installation (e.g., `brew install graphviz` on macOS, `sudo apt-get install graphviz` on Linux, or manual install on Windows).")

