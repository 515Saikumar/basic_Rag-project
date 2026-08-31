# Basic RAG Project

A robust and simple Retrieval-Augmented Generation (RAG) implementation that allows you to chat with your local documents. This project uses open-source embedding models, a local FAISS vector database, and leverages the blazing-fast Groq API for LLM inference.

## 🚀 Features

- **Multi-Format Document Loading:** Automatically processes and extracts text from various file formats (PDFs, Word documents, Text files, CSVs, Excel files, and JSON).
- **Local Vector Database:** Uses `FAISS` and `sentence-transformers` (`all-MiniLM-L6-v2`) to chunk, embed, and store document data entirely locally for fast semantic search.
- **Dynamic Groq Model Selection:** Automatically detects and selects the best active Large Language Model (Llama 3, Mixtral, Gemma, etc.) available to your specific Groq API tier, preventing annoying deprecation errors!
- **Reasoning Filtering:** Automatically strips out `<think>...</think>` reasoning blocks (from models like DeepSeek) to give you clean, direct answers.

## 📋 Prerequisites

- Python 3.8+
- A [Groq API Key](https://console.groq.com/keys)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/515Saikumar/basic_Rag-project.git
   cd basic_Rag-project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root of the project and add your Groq API key:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```

## 📂 Project Structure

```
basic_Rag-project/
├── app.py                 # Main entry point to run the application
├── requirements.txt       # Project dependencies
├── .env                   # Environment variables (not tracked by git)
├── .gitignore             # Git ignore file
└── src/
    ├── data/              # Put your source documents (PDFs, TXTs, etc.) here
    ├── data_loader.py     # Logic for reading and parsing documents
    ├── embedding.py       # Handles chunking and vector embeddings
    ├── search.py          # Groq LLM integration and final answer generation
    └── vectorstore.py     # FAISS vector database logic
```

## 💻 Usage

1. Place the documents you want to query inside the `src/data/` folder.
2. Run the main script to process the documents and generate a summary for your hardcoded query:
   ```bash
   python app.py
   ```
3. The app will build a local `faiss_store` index (so subsequent runs are much faster!) and output the LLM's summary.

## 📝 License

This project is open-source and available under the MIT License.
