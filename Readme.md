# AgenticAI LangGraph Demo

This project demonstrates a multi-modal conversational agent using [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain) with support for text, code, and vision (image) queries. The agent routes user input to the appropriate model (text, code, or vision) and can process both local and web images.

## Features

- **Text, Code, and Vision Routing:** Automatically detects the type of query and routes it to the correct model.
- **Image Support:** Accepts both local image file paths and web image URLs, encodes them, and sends them to a vision-capable LLM.
- **Interactive CLI:** Chat with the agent in your terminal.
- **Extensible Graph Structure:** Easily add new nodes or routing logic.

## Requirements

- Python 3.8+
- [Ollama](https://ollama.com/) (for running LLMs locally)
- Required Python packages (see [requirements.txt](requirements.txt))

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
    - Make sure you have [Ollama](https://ollama.com/) installed and running.
    - Download the required models:
        ```sh
        ollama pull llama3.2
        ollama pull gemma3
        ollama pull qwen2.5-coder:7b
        ```

4. **Run the agent:**
    ```sh
    python 5.FullyImplementedSolution.py
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

Type `exit` to quit the application.

## File Structure

- [`5.FullyImplementedSolution.py`](5.FullyImplementedSolution.py): Main implementation of the agentic graph.
- [`requirements.txt`](requirements.txt): Python dependencies.
- Other `.ipynb` and `.py` files: Experiments and earlier versions.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE)
