from typing import TypedDict, List, Union

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
    """
    messages: List[Union[HumanMessage, AIMessage]]


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


# ---- BUILD THE LANGGRAPH WORKFLOW ----
workflow = StateGraph(GraphState)

# Add the text node to the workflow
workflow.add_node("text_handler", text_node)

# Set the text_handler as the entry point
workflow.set_entry_point("text_handler")

# Add an edge from the text_handler to END, meaning the graph terminates after this node
workflow.add_edge("text_handler", END)

# Compile the graph to create a runnable Langchain agent
graph = workflow.compile()


# ---- TEST CASE ----
def run_test(name: str, input_messages: List[HumanMessage]):
    """
    Runs a test case for the Langgraph workflow.
    """
    print(f"\n--- Running Test: {name} ---")
    # Prepare the initial state for the graph invocation
    input_state = {
        "messages": input_messages,
    }
    
    # Invoke the graph with the input state
    result = graph.invoke(input_state)
    
    # Print the agent's final response from the last message in the state
    print(f"Agent Response: {result['messages'][-1].content}")
    print(f"Total messages in conversation history: {len(result['messages'])}")


# --- Test 1: Simple Text Query ---
run_test(
    "Simple Text Query",
    [HumanMessage(content="What is the capital of Canada?")],
)

