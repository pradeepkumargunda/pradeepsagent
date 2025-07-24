import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community import vectorstores
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import streamlit as st
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
# Ensure key is set
if not openai_api_key:
    raise ValueError("❌ OPENAI_API_KEY not set in environment. Please add it to your .env file or Streamlit Cloud Secrets.")

def getchain():
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        db=FAISS.load_local('gunda_vector_store', embeddings,allow_dangerous_deserialization=True
        )
        retriever = db.as_retriever(search_kwargs={'k':3},)
        qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4", api_key=openai_api_key),chain_type="stuff",retriever=retriever)
        return qa_chain


def main():
        chain = getchain()

        query = st.text_input("Ask a question about Pradeep's Resume?")
        if query:
                with st.spinner('thinking..'):
                        response = chain.invoke(query)
                        st.write(response.get("result","No response"))







if __name__ == '__main__':
        main()