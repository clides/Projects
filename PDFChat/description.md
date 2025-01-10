Python project using RAG (retrieval augmented generation) with langchain Ollama that allows users to chat with their PDF file

Key takeaways from this project:
- Using various langchain classes (document loading, retrieving information, creating vector database, embeddings)
- Using OllamaLLM
- Understand various new langchain and rag concepts
    - Embeddings convert each text chunk into a dense vector representation --> these chunks are used for similarity comparisons
    - FAISS creates index of the embeddings for all chunks --> allows fast similarity searches between user queries and indexed document chunks
    - Vectorstore retriever is used to fetch the most relevant text chunks for a given query
    - qa_chain combines the retriever and LLM and to create a QA system

**Summary of Workflow**
1. The PDF document is loaded and split into smaller chunks (e.g., sections, paragraphs).
2. Each chunk is converted into an embedding (a numerical representation of its content) using the OllamaEmbeddings model.
3. These embeddings are stored in FAISS, which is a highly efficient vector store.
4. When a user asks a question, the system converts the question into an embedding.
5. FAISS is then used to find the most similar document chunks (based on the embeddings).
6. The most relevant chunks are retrieved and passed to an LLM (Large Language Model, in this case, OllamaLLM) to generate an answer.