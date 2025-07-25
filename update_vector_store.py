import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, WebBaseLoader, TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from openai.types import vector_store

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("❌ OPENAI_API_KEY not found. Set it in your .env file or Streamlit Cloud secrets.")


def load_documents_pdf(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        if(filename.endswith(".pdf")):
            file_path = os.path.join(folder_path, filename)
            docs.extend(PyPDFLoader(file_path).load())
    return docs
def load_documents_word(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        if(filename.endswith(".doc") or filename.endswith(".docx")):
            file_path = os.path.join(folder_path, filename)
            docs.extend(UnstructuredWordDocumentLoader(file_path).load())
    return docs
def load_documents_txt(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        if(filename.endswith(".txt")):
            file_path = os.path.join(folder_path, filename)
            docs.extend(TextLoader(file_path).load())
    return docs
def update_vector_store(docs,store_path='gunda_vector_store'):

    # Load existing vector store
    vector_store = FAISS.load_local(store_path,OpenAIEmbeddings(api_key=openai_api_key), allow_dangerous_deserialization=True
    )

    splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    # Embed and add to store
    print(chunks)
    vector_store.add_documents(chunks)
    vector_store.save_local('gunda_vector_store')



    # Save updated store











if __name__ == "__main__":
    documents = load_documents_pdf('new_data')
    if(len(documents) != 0):
        update_vector_store(documents)

    documents = load_documents_word('new_data')
    if(len(documents) != 0):
        print(documents)
        #update_vector_store(documents)

    documents = load_documents_txt('new_data')
    if (len(documents) != 0):
        update_vector_store(documents)

    #urls=["https://www.linkedin.com/in/pradeepkgunda/"]
    #web_loader = WebBaseLoader(urls)
    #web_docs = web_loader.load()
    #update_vector_store(documents)

