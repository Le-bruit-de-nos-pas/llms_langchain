import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp  # Local LLM

# Load local embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_k_relevant_documents(documents, question, k=1):
    print(f"Storing {len(documents)} into Vector Store.")
    
    # Store documents in ChromaDB (local vector database)
    vector_store = Chroma.from_documents(documents, embedding_model)
    
    print("Getting relevant documents from local vector store.")
    relevant_docs = vector_store.similarity_search(question, k=k)
    print(f"Retrieved similar documents: {len(relevant_docs)}")
    
    return relevant_docs

def get_answer_from_llm(documents, question):
    print(f"Question: {question}")
    
    relevant_docs = get_k_relevant_documents(documents, question)   
    context_from_docs = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Load LLaMA model locally
    llm = LlamaCpp(
        model_path="e5-mistral-7b-instruct-Q6_K.gguf",  # Replace with your model path
        temperature=0.7,
        max_tokens=3072,
        n_ctx=3072,
    )

    messages = [
        SystemMessage(content=f"Use the following context to answer my question: {context_from_docs}"),
        HumanMessage(content=question),
    ]

    parser = StrOutputParser()
    chain = llm | parser
    return chain.invoke(messages)
