import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community import vectorstores
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import streamlit as st
load_dotenv()

def getchain():
        db=FAISS.load_local('gunda_vector_store', OpenAIEmbeddings(),allow_dangerous_deserialization=True
        )
        retriever = db.as_retriever(search_kwargs={'k':3},)
        qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4"),chain_type="stuff",retriever=retriever)
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