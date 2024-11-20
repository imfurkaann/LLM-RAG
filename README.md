# Document Analysis and QA Bot

This project is a document-based Question-Answering (QA) bot that enables users to upload files (CSV, TXT, PDF) and ask questions based on the uploaded content. It leverages **LangChain**, **OpenAI**, and **Gradio** to process the documents, extract insights, and respond to user queries.

## Features
- **File Upload**: Supports CSV, TXT, and PDF formats.
- **Dynamic QA**: Provides answers to user queries using the uploaded document's content.
- **Embeddings with OpenAI**: Converts document text into embeddings for semantic understanding.
- **Persistent Database**: Uses **Chroma** for efficient retrieval of document information.
- **Interactive Web Interface**: Built using **Gradio** for a user-friendly experience.

---

## How It Works
1. **File Upload**: Users upload a document (CSV, TXT, or PDF).
2. **Document Processing**: The document is read and processed into text using:
   - **CSVLoader** for CSV files.
   - **TextLoader** for TXT and PDF files (uses LangChain's TextLoader).
3. **Embeddings Generation**: OpenAI embeddings are created for the document text, enabling efficient semantic search.
4. **Question Processing**: User questions are matched against the document using **Chroma**'s retriever and processed with OpenAI.
5. **Answer Generation**: A custom prompt guides the model to produce accurate and context-aware answers.

---

## Technologies Used
- **LangChain**: For document loading, processing, and chaining LLM queries.
- **OpenAI**: Generates embeddings and processes user queries.
- **Chroma**: Persistent vector database for document retrieval.
- **Gradio**: Provides a simple and interactive web interface.

---

## Setup Instructions
### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/imfurkaann/LLM-RAG.git
   cd document-analysis-qa-bot
   ```
2. Create a virtual environment and activate it:
   ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies
   ```bash
    pip install -r req.txt
   ```
## Run the Application
Start the Gradio app: 
``` bash
  python llm_rag.py
```


