import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter

# split document into chunks
def setup_qa_system(file_path):
    # loading the pdf document
    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split() # splitting the pdf into seperate pieces
    
    # splitting the document
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    
    # creating the embeddings model
    embeddings = OllamaEmbeddings(model='openhermes',
                                  base_url="http://localhost:11434")
    vectorstore = FAISS.from_documents(chunks, embeddings) # creating vector store
    
    # retrieves the chunks that are the most relevant
    retriever = vectorstore.as_retriever()
    llm = OllamaLLM(model='openhermes', base_url="http://localhost:11434")
    
    # use llm to give answer
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    
    return qa_chain

if __name__ == "__main__":
    qa_chain = setup_qa_system("resume.pdf")
    
    while True:
        question = input("\nEnter your question (type 'exit' to quit): ")
        if question.lower() == "exit":
            break
        
        answer = qa_chain.invoke(question)
        print('Answer:')
        print(answer['result'])