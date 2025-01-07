Python project using RAG (retrieval augmented generation) with langchain Ollama that allows users to chat with their PDF file

Key takeaways from this project:
- Using various langchain classes (document loading, retrieving information, creating vector database, embeddings)
- Using OllamaLLM
- Understand various new langchain and rag concepts
    - Embeddings convert each text chunk into a dense vector representation --> these chunks are used for similarity comparisons
    - FAISS creates index of the embeddings for all chunks --> allows fast similarity searches between user queries and indexed document chunks
    - Vectorstore retriever is used to fetch the most relevant text chunks for a given query
    - qa_chain combines the retriever and LLM and to create a QA system