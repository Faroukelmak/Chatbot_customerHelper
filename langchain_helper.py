import os

from dotenv import load_dotenv
load_dotenv()

from langchain.document_loaders import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

api_key=os.environ["Googel_Api_key"]

llm = GooglePalm(google_api_key=api_key, temperature=0.7)

# Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

vectordb_file_path = "faiss_index"

def create_vector_db():
    file_path = 'codebasics_faqs.csv'
    loader = CSVLoader(file_path=file_path, source_column='prompt', encoding='latin-1')
    data = loader.load()
    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data,
                                 embedding=instructor_embeddings)
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    vectordb=FAISS.load_local(vectordb_file_path, instructor_embeddings)
    # retriever hoa li kiya5d la question ou kichouf embedding dialha ou kijbd li b7alha f vectorstore

    retriever = vectordb.as_retriever(score_threshold=0.7)


    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs=chain_type_kwargs)
    return chain

if __name__ == "__main__":
    chain = get_qa_chain()
    print(chain("do you provide internship? Do you have emi option ?"))