import sys
import os
from typing import List, Literal, Union

# Add the parent directory to the Python path to allow importing langgraph_agent
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from langgraph_agent import compile_langgraph_workflow, GraphState
from langchain_core.messages import HumanMessage, AIMessage

# Compile the graph for testing
graph = compile_langgraph_workflow()

# Helper function to run a test and print the agent's response
def run_test(name: str, input_messages: List[HumanMessage], initial_model_choice: Literal["vision", "text", "code", "auto"] = "auto"):
    """
    Runs a single turn test case for the Langgraph workflow.
    """
    print(f"\n--- Running Test: {name} ---")
    input_state = {
        "messages": input_messages,
        "model_choice": initial_model_choice,
        "next_node_hint": "text_handler" # Default, will be overridden by preprocess_and_route_node
    }
    
    result = graph.invoke(input_state)
    
    print(f"Agent Response: {result['messages'][-1].content}")
    print(f"Total messages in conversation history: {len(result['messages'])}")


# ---- TEST CASES ----

# --- Test 1: Simple Text Query (LLM should route to text) ---
run_test(
    "Simple Text Query (LLM-routed)",
    [HumanMessage(content="What is the capital of Canada?")],
    initial_model_choice="auto"
)

# --- Test 2: Code Generation Request (LLM should route to code) ---
run_test(
    "Code Generation Request (LLM-routed)",
    [HumanMessage(content="Write a Python function to calculate factorial.")],
    initial_model_choice="auto"
)

# --- Test 3: Image Description Request (using a web URL - direct route) ---
run_test(
    "Image Description Request (web URL - direct route)",
    [HumanMessage(content="Describe the weather in this image in two lines: https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg")],
    initial_model_choice="auto"
)

# --- Test 4: Code Generation Request (explicit hint - direct route) ---
run_test(
    "Code Generation Request (explicit hint - direct route)",
    [HumanMessage(content="How do I implement a quicksort algorithm in Java?")],
    initial_model_choice="code"
)

# --- Test 5: Image Description Request (explicit hint, another web URL - direct route) ---
run_test(
    "Image Description Request (explicit hint, another web URL - direct route)",
    [HumanMessage(content="Describe this picture in two lines: https://homedecorez.co.uk/wp-content/uploads/2024/12/Practical-Magic-House.webp")],
    initial_model_choice="vision"
)

# --- Test 6: Local Image Description Request (direct route) ---
# IMPORTANT: Replace 'D:\ML\AgenticAI_Projects\AgenticAI_langgraph\R.jpeg' with an actual, valid path
run_test(
    "Local Image Description Request (direct route)",
    [HumanMessage(content=r"Analyze the content of this local image: D:\ML\AgenticAI_Projects\AgenticAI_langgraph\R.jpeg")],
    initial_model_choice="auto"
)

# --- Test 7: Ambiguous Text Query (LLM should still route to text) ---
run_test(
    "Ambiguous Text Query (LLM-routed)",
    [HumanMessage(content="Tell me a story about a brave knight.")],
    initial_model_choice="auto"
)

