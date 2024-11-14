RAG-FAISS Embedding with SQLite Pipeline
========================================

This project implements a Retrieval-Augmented Generation (RAG) system combined with FAISS for efficient information retrieval and embedding search, supported by a FastAPI server for API access. It includes vector storage using SQLite and a complete data ingestion and query pipeline.

## Features

- **Embedding Creation and Storage**: Generates vector embeddings and stores them in FAISS and SQLite for fast similarity search.
- **Data Ingestion**: Processes unstructured HTML data into a structured format for vectorization.
- **API Access**: Exposes endpoints using FastAPI for real-time search and response generation.
- **Health Check**: Includes a health check endpoint for easy monitoring.
- **Scripted Pipelines**: Bash scripts automate data processing and server startup.

## Project Structure

- **1-rag-faiss-sqlite-pipeline.sh**: Shell script for data processing and initializing the RAG datastore manager.
- **3-fastapi-uvicorn-server.sh**: Script to start the FastAPI server with Uvicorn, including a browser auto-launch for documentation.
- **config.py**: Configuration settings for the RAG and FAISS system.
- **data_ingestion.py**: Script for processing and ingesting raw data into the system.
- **database.py**: Manages interactions with the SQLite database for storing metadata.
- **faiss_store.py**: Handles the FAISS index creation and storage of vector embeddings.
- **query.py**: Main FastAPI application for search queries.
- **vectorization.py**: Contains functions for vectorizing text data using sentence-transformers.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/luzbetak/rag-faiss-embedding.git
   cd rag-faiss-embedding
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure FAISS and SQLite are properly installed.

## Usage

1. Run the data ingestion and setup pipeline:
   ```bash
   ./1-rag-faiss-sqlite-pipeline.sh
   ```

2. Start the FastAPI server:
   ```bash
   ./3-fastapi-uvicorn-server.sh
   ```

   - Access the API documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

## API Endpoints

- **POST /search**: Search and generate responses.
- **GET /health**: Health check endpoint.

## Contributing

Contributions are welcome! Fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the [MIT License](LICENSE).
