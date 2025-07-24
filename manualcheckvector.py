import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import streamlit as st

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vector_db = FAISS.load_local('gunda_vector_store', embeddings=embeddings, allow_dangerous_deserialization=True)
docs= vector_db.similarity_search("What is the email ID of Pradeep")
print(docs)