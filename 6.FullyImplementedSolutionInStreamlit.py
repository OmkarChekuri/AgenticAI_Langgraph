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
        "llm_vision": ChatOllama(model="gemma3"), # Retaining 'gemma3' as per your confirmation
        "llm_code": ChatOllama(model="qwen2.5-coder:7b")
    }

llm_models = load_llm_models()
llm_text = llm_models["llm_text"]
llm_vision = llm_models["llm_vision"]
llm_code = llm_models["llm_code"]


# ---- HELPER FUNCTIONS ----
def encode_image_to_base64(image_bytes: bytes) -> str:
    """
    Encodes image bytes (from an uploaded file) to a base64 string.
    """
    return base64.b64encode(image_bytes).decode('utf-8')

def fetch_and_encode_web_image(image_url: str) -> Union[str, None]:
    """
    Fetches an image from a URL and encodes it to a base64 string.
    """
    print(f"Fetching and encoding web image from URL: {image_url}")
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
    #print(f"Vision Node - Incoming Messages: {state['messages']}") # Commented out for cleaner output
    try:
        response = llm_vision.invoke(state["messages"])
        if response and response.content:
            #print("response from user:", response.content)  # Print the response content for debugging
            state["messages"].append(AIMessage(content=response.content))
        else:
            # If response is empty or None, provide a fallback message
            state["messages"].append(AIMessage(content="Vision model did not provide a clear response. It might be struggling to interpret the image or the query."))
            print("Warning: Vision model returned empty or no content.")
    except Exception as e:
        # Catch any exceptions during vision model invocation
        error_message = f"An error occurred while processing the image with the vision model: {e}. Please ensure the model is running and compatible with the image format."
        state["messages"].append(AIMessage(content=error_message))
        st.error(error_message) # Display error in Streamlit UI
        print(f"Error in vision_node: {e}")
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
    st.session_state.current_llm_hint = state['next_node_hint'] # Update hint immediately
    return state


# ---- PREPROCESSING AND ROUTING NODE (Simplified for Streamlit Integration) ----
def preprocess_and_route_node(state: GraphState) -> GraphState:
    print("---Entering PREPROCESS AND ROUTE NODE---")
    last_msg = state["messages"][-1]
    
    if "model_choice" not in state:
        state["model_choice"] = "auto"

    next_hint_from_preprocess = None
    
    # Check if the message is already multimodal (meaning image was processed by UI logic)
    # This is the primary way image inputs are identified in the Langgraph workflow now.
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
            # If no explicit model_choice AND no image was pre-processed by UI,
            # then delegate to the LLM router for text/code keyword classification.
            next_hint_from_preprocess = "llm_router_node"
            if isinstance(last_msg, HumanMessage) and isinstance(last_msg.content, str):
                code_keywords = ["code", "program", "function", "script", "develop", "implement", "write in",
                                 "python", "javascript", "java", "c++", "html", "css", "sql", "bash"]
                if any(keyword in last_msg.content.lower() for keyword in code_keywords):
                    print("Code keywords detected, routing to LLM Router for decision.")


    state["next_node_hint"] = next_hint_from_preprocess
    print(f"Preprocess node decided next hint: {next_hint_from_preprocess}")
    st.session_state.current_llm_hint = state['next_node_hint'] # Update hint immediately
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
# Temporary storage for image data uploaded/detected in the current turn, awaiting text prompt
if "pending_image_data" not in st.session_state:
    st.session_state.pending_image_data = None
if "pending_image_mime_type" not in st.session_state:
    st.session_state.pending_image_mime_type = None
if "pending_image_display_text" not in st.session_state:
    st.session_state.pending_image_display_text = None
if "current_llm_hint" not in st.session_state:
    st.session_state.current_llm_hint = "Waiting for input..."
# Key for file uploader to allow resetting it
if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0


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
                                    st.image(image_url_data, caption="User provided image", width=200)
                                else:
                                    st.markdown(f"Image URL: {image_url_data}") # Fallback for non-data URLs
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

# --- Image Upload ---
# Use a key to allow programmatic resetting of the uploader
uploaded_file = st.file_uploader("Upload an image:", type=["png", "jpg", "jpeg", "gif", "bmp", "tiff"], key=f"file_uploader_{st.session_state.file_uploader_key}")

# --- Process Uploaded File ---
if uploaded_file is not None and st.session_state.pending_image_data is None:
    # Only process if a new file is uploaded and no image is currently pending
    image_bytes = uploaded_file.read()
    st.session_state.pending_image_data = encode_image_to_base64(image_bytes)
    st.session_state.pending_image_mime_type = uploaded_file.type
    st.session_state.pending_image_display_text = f"Uploaded Image: {uploaded_file.name}"
    st.session_state.messages.append(HumanMessage(content=f"Image uploaded. Please type your query and press Enter."))
    st.rerun() # Rerun to show the "Image uploaded" message and clear the uploader widget visually

# --- User Input Area ---
user_query = st.chat_input("Type your message here...")

# Display the current LLM hint
st.info(f"Current LLM: {st.session_state.current_llm_hint.replace('_handler', '').replace('_node', '').title()}")


# --- Main Processing Logic (triggered by user_query submission) ---
if user_query:
    final_human_message_content = user_query
    image_source_display_text = None

    # 1. Check for pending uploaded image
    if st.session_state.pending_image_data:
        final_human_message_content = [
            {"type": "text", "text": user_query},
            {"type": "image_url", "image_url": {"url": f"data:{st.session_state.pending_image_mime_type};base64,{st.session_state.pending_image_data}"}},
        ]
        image_source_display_text = st.session_state.pending_image_display_text
        # Clear pending image data after it's used
        st.session_state.pending_image_data = None
        st.session_state.pending_image_mime_type = None
        st.session_state.pending_image_display_text = None
        # Increment uploader key to visually reset it
        st.session_state.file_uploader_key += 1
        
    else: # 2. Check for web/local image URLs in text query
        text_content_lower = user_query.lower()
        
        # --- FIX: Corrected regex for web image URLs to include http:// ---
        web_image_url_match = re.search(r'(https?:\/\/[^\s\/$.?#].[^\s]*?\.(jpg|jpeg|png|gif|bmp|svg|tiff))', text_content_lower, re.IGNORECASE)
        if web_image_url_match:
            detected_url = web_image_url_match.group(1)
            image_extension = web_image_url_match.group(2)
            st.info(f"Detected web image URL: {detected_url}")
            base64_img = fetch_and_encode_web_image(detected_url)
            if base64_img:
                final_human_message_content = [
                    {"type": "text", "text": user_query.replace(web_image_url_match.group(0), "").strip() + ", also don't infer anything from the path name" or "Describe this image, also don't infer anything from the path name:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/{image_extension};base64,{base64_img}"}},
                ]
                image_source_display_text = f"Web Image: {detected_url}"
            else:
                st.warning("Failed to fetch or encode web image.")
                final_human_message_content = user_query
                image_source_display_text = None
        else: # Only check for local file paths if no web URL was found
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
                text_content_lower,
                re.IGNORECASE
            )

            if local_file_path_match:
                detected_path = local_file_path_match.group(2)
                image_extension = local_file_path_match.group(3)
                st.info(f"Detected local image path: {detected_path}")
                try:
                    image_bytes_from_path = open(detected_path, "rb").read()
                    base64_img = encode_image_to_base64(image_bytes_from_path)
                    final_human_message_content = [
                        {"type": "text", "text": user_query.replace(local_file_path_match.group(0), "").strip() + ", also, don't infer anything from the path name" or "Describe this image, also, don't infer anything from the path name:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/{image_extension};base64,{base64_img}"}},
                    ]
                    image_source_display_text = f"Local Image: {detected_path}"
                except FileNotFoundError:
                    st.error(f"Local image file not found: {detected_path}. Please ensure the path is correct and accessible.")
                    final_human_message_content = user_query
                    image_source_display_text = None
                except Exception as e:
                    st.error(f"Error processing local image: {e}")
                    final_human_message_content = user_query
                    image_source_display_text = None



    # Add the user's message (potentially multimodal) to display history
    st.session_state.messages.append(HumanMessage(content=final_human_message_content))
    if image_source_display_text:
        # Optionally, add a small info message about the image source below the user's message
        with chat_container: # Re-enter container to append
            with st.chat_message("user"):
                st.caption(image_source_display_text)


    # Update the Langgraph conversation state with the new user message
    st.session_state.conversation_state["messages"].append(HumanMessage(content=final_human_message_content))

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
                st.session_state.conversation_state["messages"].pop()
                
    st.rerun() # Rerun to update chat display


# --- Clear Chat Button ---
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.conversation_state = {
        "messages": [],
        "model_choice": "auto",
        "next_node_hint": "text_handler"
    }
    st.session_state.pending_image_data = None
    st.session_state.pending_image_mime_type = None
    st.session_state.pending_image_display_text = None
    st.session_state.current_llm_hint = "Waiting for input..."
    st.session_state.file_uploader_key += 1 # Increment key to reset file uploader visually
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
