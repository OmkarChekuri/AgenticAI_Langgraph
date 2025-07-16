# AgenticAI LangGraph Streamlit App

This project is a multi-modal conversational agent built with [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://github.com/langchain-ai/langchain), and [Streamlit](https://streamlit.io/). It routes user queries to specialized LLMs for text, code, or vision (image) tasks, and supports both image uploads and URLs.

## Features

- **Dynamic Routing:** Automatically detects if your query is text, code, or image and routes it to the appropriate model.
- **Image Support:** Upload images or provide image URLs (local or web) for vision-based queries.
- **Interactive Web UI:** Powered by Streamlit for a modern chat experience.
- **Workflow Visualization:** View the LangGraph workflow graph in the sidebar.

## Requirements

- Python 3.8+
- [Ollama](https://ollama.com/) (for running LLMs locally)
- Required Python packages (see `requirements.txt`)

## Setup

1. **Clone the repository:**
    ```sh
    git clone <your-repo-url>
    cd AgenticAI_langgraph
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Download Ollama models:**
    ```sh
    ollama pull llama3.2
    ollama pull gemma3
    ollama pull qwen2.5-coder:7b
    ```

4. **Install Graphviz (for workflow visualization):**
    - **Windows:** Download and install from [Graphviz website](https://graphviz.gitlab.io/download/).
    - **macOS:** `brew install graphviz`
    - **Linux:** `sudo apt-get install graphviz`

5. **Run the Streamlit app:**
    ```sh
    streamlit run 6.FullyImplementedSolutionInStreamlit.py
    ```

## Usage

- **Text queries:**  
  `What is the capital of France?`
- **Code queries:**  
  `Write a Python function to reverse a string.`
- **Local image:**  
  `Describe this image: C:\Users\YourUser\Pictures\my_image.png`
- **Web image:**  
  `What's in this picture? https://example.com/image.jpg`
- **Image upload:**  
  Use the file uploader in the UI.

Type your message in the chat input or upload an image. The agent will respond accordingly.

## File Structure

- `6.FullyImplementedSolutionInStreamlit.py` — Main Streamlit app.
- `requirements.txt` — Python dependencies.
- Other `.py` and `.ipynb` files — Experiments and earlier versions.

## License

This project is licensed under the MIT License.