from langchain_community.document_loaders import PyPDFLoader
from get_relevant_documents import get_answer_from_llm

pdf_file_path = "./papers/Visual preference for social stimuli in individuals with autism or neurodevelopmental disorders.pdf"

loader = PyPDFLoader(pdf_file_path)

documents = list(loader.lazy_load())  # Load all pages into a list

answer = get_answer_from_llm(
    documents=documents,
    question="What was the study population for this scientific study/paper? Return the answer as 1 single sentence."
)

print("---------------FINAL ANSWER-----------------------")
print(answer)
