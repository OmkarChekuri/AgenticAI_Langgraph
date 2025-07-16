import streamlit as st
import re
import io # For graph image bytes
import base64 # For image encoding in UI
import os # Import os for path checking

# Import the Langgraph agent components and the llm_models dictionary
from langgraph_agent import (
    GraphState, # Import GraphState for type hinting if needed, though not strictly used in UI logic
    compile_langgraph_workflow,
    encode_image_to_base64,
    fetch_and_encode_web_image,
    llm_models # Import llm_models dictionary
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
    elif base_name.endswith("_error") or base_name.endswith("_fallback"): # Handle error/fallback hints
        # Extract the original handler name, e.g., "vision_handler_error" -> "vision"
        original_handler = base_name.split('_')[0]
        if original_handler == "vision":
            model_id = llm_models["llm_vision"].model
        elif original_handler == "text":
            model_id = llm_models["llm_text"].model
        elif original_handler == "code":
            model_id = llm_models["llm_code"].model
        
        display_name = f"{original_handler.title()} (Error/Fallback)" # More descriptive
    
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
# Session state for graph visibility
if "show_graph" not in st.session_state:
    st.session_state.show_graph = True # Default to True to show graph by default
# NEW: Session state for explicit model selection
if "selected_model_type" not in st.session_state:
    st.session_state.selected_model_type = "Auto-route"


# --- Display Chat Messages ---
chat_container = st.container(height=250, border=True)
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
                                    st.image(image_url_data, caption="Uploaded Image", width=100)
                                else:
                                    st.markdown(f"Image URL: {image_url_data}") # Fallback for non-data URLs
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                # Display LLM name next to bot icon
                llm_hint_for_display = message.additional_kwargs.get("llm_hint")
                if llm_hint_for_display:
                    display_llm_name = get_display_llm_name(llm_hint_for_display)
                    st.markdown(f"**{display_llm_name}**") # Display the LLM name
                st.markdown(message.content) # Display the actual response

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
st.info(f"Current LLM: {get_display_llm_name(st.session_state.current_llm_hint)}")


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

    # Set the model_choice based on radio button selection
    if st.session_state.selected_model_type == "Auto-route":
        st.session_state.conversation_state["model_choice"] = "auto"
    elif st.session_state.selected_model_type == "Text Model":
        st.session_state.conversation_state["model_choice"] = "text"
    elif st.session_state.selected_model_type == "Vision Model":
        st.session_state.conversation_state["model_choice"] = "vision"
    elif st.session_state.selected_model_type == "Code Model":
        st.session_state.conversation_state["model_choice"] = "code"


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
    st.session_state.selected_model_type = "Auto-route" # Reset radio button selection
    st.rerun()

# --- Workflow Graph Visualization (in sidebar) ---
st.sidebar.header("Workflow Graph")

# Toggle button for graph visibility
if st.sidebar.button("Hide Graph" if st.session_state.show_graph else "Show Graph"):
    st.session_state.show_graph = not st.session_state.show_graph
    st.rerun() # Rerun to apply visibility change

if st.session_state.show_graph:
    # Define the path where the graph image is saved
    graph_file_path = "langgraph_workflow.png"
    
    # Check if the file exists before trying to display it
    if os.path.exists(graph_file_path):
        # Use use_container_width instead of use_column_width
        st.sidebar.image(graph_file_path, caption="Langgraph Workflow", use_container_width=True)
    else:
        st.sidebar.warning(f"Graph visualization not found at '{graph_file_path}'.")
        


# --- Model Selection Radio Button (NEW) ---
st.sidebar.markdown("---") # Separator
st.sidebar.subheader("Force Model Selection")
selected_option = st.sidebar.radio(
    "Choose routing behavior:",
    ("Auto-route", "Text Model", "Vision Model", "Code Model"),
    key="model_selection_radio",
    index=("Auto-route", "Text Model", "Vision Model", "Code Model").index(st.session_state.selected_model_type)
)
if selected_option != st.session_state.selected_model_type:
    st.session_state.selected_model_type = selected_option
    st.rerun() # Rerun if selection changes to update model_choice immediately

# --- List Loaded Models ---
st.sidebar.markdown("---") # Separator
st.sidebar.subheader("Loaded LLM Models")
for llm_key, llm_instance in llm_models.items():
    model_type = llm_key.replace("llm_", "").title()
    st.sidebar.markdown(f"- **{model_type}**: `{llm_instance.model}`")