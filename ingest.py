import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("❌ OPENAI_API_KEY not found. Set it in your .env file or Streamlit Cloud secrets.")


def load_documents(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        if(filename.endswith(".pdf")):
            file_path = os.path.join(folder_path, filename)
            docs.extend(PyPDFLoader(file_path).load())
    return docs

def split_and_embed(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local('gunda_vector_store')







if __name__ == "__main__":
    documents = load_documents('data')
    split_and_embed(documents)