from typing import TypedDict, List, Literal, Union

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
                      This is added for future routing logic.
        next_node_hint: A hint set by the preprocess_and_route_node to guide the next transition.
                        This is crucial for the conditional routing.
    """
    messages: List[Union[HumanMessage, AIMessage]]
    model_choice: Literal["vision", "text", "code", "auto"] # Added for future routing
    next_node_hint: Literal["vision_handler", "text_handler", "code_handler"] # Added for routing decision


# ---- MODEL INITIALIZATION ----
# Initialize a text-capable Ollama model.
# Ensure 'llama3.2' is downloaded via Ollama: ollama pull llama3.2
llm_text = ChatOllama(model="llama3.2")


# ---- GRAPH NODES ----
def text_node(state: GraphState) -> GraphState:
    """
    Processes the message using the text-capable LLM.
    Appends the AI's response to the messages list.
    """
    print("---Entering TEXT NODE---")
    # Invoke the text model with the current conversation history
    response = llm_text.invoke(state["messages"])
    # Append the AI's response to the state's messages
    state["messages"].append(AIMessage(content=response.content))
    return state


# ---- PREPROCESSING AND ROUTING NODE (New) ----
def preprocess_and_route_node(state: GraphState) -> GraphState:
    """
    This node will eventually preprocess the input (e.g., detect image paths)
    and set a hint in the state for the next routing decision.
    For now, it simply sets the hint to direct to the text_handler.
    """
    print("---Entering PREPROCESS AND ROUTE NODE---")
    # Initialize model_choice if it's not present in the state (for the first turn)
    if "model_choice" not in state:
        state["model_choice"] = "auto"
    
    # For this step, we always set the hint to go to the text_handler
    state["next_node_hint"] = "text_handler"
    print(f"Preprocess node decided next hint: {state['next_node_hint']}")
    return state # Important: A node must return the updated state


# ---- ROUTING DECISION FUNCTION (New) ----
# This function is used by Langgraph's add_conditional_edges to decide the next node.
# It simply reads the hint set by the preprocess_and_route_node.
def route_decision(state: GraphState) -> Literal["vision_handler", "text_handler", "code_handler"]:
    """
    Reads the 'next_node_hint' from the state to determine the next node.
    This function is used by add_conditional_edges and only returns a string.
    """
    print(f"---Making ROUTING DECISION based on hint: {state['next_node_hint']}---")
    return state["next_node_hint"]


# ---- BUILD THE LANGGRAPH WORKFLOW ----
workflow = StateGraph(GraphState)

# Add the new preprocess_and_route_node
workflow.add_node("preprocess_and_route_node", preprocess_and_route_node)
# Keep the existing text_handler node
workflow.add_node("text_handler", text_node)

# Set the preprocess_and_route_node as the entry point of the graph
# All interactions will now begin by passing through this node.
workflow.set_entry_point("preprocess_and_route_node")

# Add conditional edges from the preprocess_and_route_node
# The 'route_decision' function will determine the next node.
workflow.add_conditional_edges(
    "preprocess_and_route_node", # Source node for the edges
    route_decision,              # The function that decides the next step based on state
    {
        "text_handler": "text_handler",   # If route_decision returns "text_handler", go to text_handler node
        # We will add "vision_handler" and "code_handler" mappings in later steps
    }
)

# The text_handler still goes to END for a single turn
workflow.add_edge("text_handler", END)

# Compile the graph
graph = workflow.compile()


# ---- TEST CASE ----
def run_test(name: str, input_messages: List[HumanMessage], initial_model_choice: Literal["vision", "text", "code", "auto"] = "auto"):
    """
    Runs a test case for the Langgraph workflow.
    """
    print(f"\n--- Running Test: {name} ---")
    # Prepare the initial state for the graph invocation
    input_state = {
        "messages": input_messages,
        "model_choice": initial_model_choice,
        # next_node_hint is initialized here, but preprocess_and_route_node will set it
        "next_node_hint": "text_handler" 
    }
    
    # Invoke the graph with the input state
    result = graph.invoke(input_state)
    
    # Print the agent's final response from the last message in the state
    print(f"Agent Response: {result['messages'][-1].content}")
    print(f"Total messages in conversation history: {len(result['messages'])}")


# --- Test 1: Simple Text Query (now routed) ---
run_test(
    "Simple Text Query via Router",
    [HumanMessage(content="What is the capital of Canada?")],
    initial_model_choice="auto"
)
