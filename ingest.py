
import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, WebBaseLoader, \
        TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from openai.types import vector_store

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

if not openai_api_key:
    raise ValueError("❌ OPENAI_API_KEY not found. Set it in your .env file or Streamlit Cloud secrets.")


def load_documents_pdf(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        if (filename.endswith(".pdf")):
                file_path = os.path.join(folder_path, filename)
                docs.extend(PyPDFLoader(file_path).load())
    return docs


def load_documents_word(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        if (filename.endswith(".doc") or filename.endswith(".docx")):
            file_path = os.path.join(folder_path, filename)
            docs.extend(UnstructuredWordDocumentLoader(file_path).load())
    return docs


def load_documents_txt(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        if (filename.endswith(".txt")):
            file_path = os.path.join(folder_path, filename)
            docs.extend(TextLoader(file_path).load())
    return docs


def update_vector_store(docs, store_path='gunda_vector_store'):

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Embed and add to store
    for chunk in chunks:
        print(chunk)
        print("\n")
    vector_store.add_documents(chunks)
    vector_store.save_local('gunda_vector_store')


        # Save updated store


if __name__ == "__main__":
    documents = load_documents_pdf('data')
    if (len(documents) != 0):
        update_vector_store(documents)

    documents = load_documents_txt('data')
    if (len(documents) != 0):
        update_vector_store(documents)
