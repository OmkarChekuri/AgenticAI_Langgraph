import streamlit as st
import re
import io # For graph image bytes
import base64 # For image encoding in UI

# Import the Langgraph agent components and the llm_models dictionary
from langgraph_agent import (
    GraphState, # Import GraphState for type hinting if needed, though not strictly used in UI logic
    compile_langgraph_workflow,
    encode_image_to_base64,
    fetch_and_encode_web_image,
    llm_models # <--- NEW: Import llm_models dictionary
)
from langchain_core.messages import HumanMessage, AIMessage # For message types


# Compile the Langgraph workflow (cached)
graph = compile_langgraph_workflow()

# --- Helper function to get display name for LLM ---
def get_display_llm_name(hint: str) -> str:
    """
    Maps the internal node hint to a user-friendly LLM name with its model ID.
    """
    base_name = hint.replace('_handler', '').replace('_node', '')
    display_name = base_name.title()
    model_id = ""

    if base_name == "text":
        model_id = llm_models["llm_text"].model
    elif base_name == "vision":
        model_id = llm_models["llm_vision"].model
    elif base_name == "code":
        model_id = llm_models["llm_code"].model
    elif base_name == "llm_router": # The router itself uses llm_text
         model_id = llm_models["llm_text"].model
    
    if model_id:
        return f"{display_name} ({model_id})"
    return display_name


# --- 2. STREAMLIT APPLICATION UI ---

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
                                    st.image(image_url_data, caption="Uploaded Image", width=200)
                                else:
                                    st.markdown(f"Image URL: {image_url_data}") # Fallback for non-data URLs
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

# --- Image Upload ---
# Use a key to allow programmatic resetting of the uploader
uploaded_file = st.file_uploader("Upload an image:", type=["png", "jpg", "jpeg", "gif", "bmp", "tiff", "svg"], key=f"file_uploader_{st.session_state.file_uploader_key}")

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
st.info(f"Current LLM: {get_display_llm_name(st.session_state.current_llm_hint)}") # <--- UPDATED LINE


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
        
        # Prioritize WEB image URLs
        web_image_url_match = re.search(r'(https?:\/\/[^\s]+\.(jpg|jpeg|png|gif|bmp|tiff|svg))', text_content_lower, re.IGNORECASE)

        if web_image_url_match:
            detected_url = web_image_url_match.group(1)
            image_extension = web_image_url_match.group(2)
            st.info(f"Detected web image URL: {detected_url}")
            base64_img = fetch_and_encode_web_image(detected_url)
            if base64_img:
                clean_text_prompt = user_query.replace(web_image_url_match.group(0), "").strip()
                final_human_message_content = [
                    {"type": "text", "text": clean_text_prompt or "Describe this image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/{image_extension};base64,{base64_img}"}},
                ]
                image_source_display_text = f"Web Image: {detected_url}"
            else:
                st.warning("Failed to fetch or encode web image.")
                final_human_message_content = user_query
                image_source_display_text = None
        else: # Check for local file paths if no web URL was found
            local_file_path_match = re.search(
                r'(file://)?'                                 
                r'('                                           
                r'(?:[a-zA-Z]:[\\/]|[\/])'                     
                r'(?:[^<>:"|?*\\/]+\\?)*'                  
                r'[^<>:"|?*\\/]*'                             
                r'\.'                                         
                r'(jpg|jpeg|png|gif|bmp|tiff|svg)'                
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
                    clean_text_prompt = user_query.replace(local_file_path_match.group(0), "").strip()
                    final_human_message_content = [
                        {"type": "text", "text": clean_text_prompt or "Describe this image:"},
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
