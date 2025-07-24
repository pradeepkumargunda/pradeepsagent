import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import streamlit as st

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("❌ OPENAI_API_KEY not set in environment. Please add it to your .env file or Streamlit Cloud Secrets.")

# Set Streamlit page config
st.set_page_config(
    page_title="📄 Resume Chatbot - Pradeep",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Define styles with markdown
st.markdown("""
    <style>
        .main-title {
            font-size: 2.5em;
            font-weight: 700;
            text-align: center;
            margin-bottom: 0.3em;
        }
        .subheader {
            text-align: center;
            font-size: 1.1em;
            color: gray;
            margin-bottom: 1.5em;
        }
        .response-box {
            background-color: #f8f9fa;
            border-radius: 0.75rem;
            padding: 1.25rem;
            margin-top: 1rem;
            font-size: 1.1em;
            box-shadow: 0 4px 8px rgba(0,0,0,0.08);
        }
        footer {
            margin-top: 4rem;
            text-align: center;
            color: #aaa;
            font-size: 0.9em;
        }
    </style>
""", unsafe_allow_html=True)

# Chain creation function
def getchain():
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.load_local('gunda_vector_store', embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={'k': 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4", api_key=openai_api_key),
        chain_type="stuff",
        retriever=retriever
    )
    return qa_chain

# Main app
def main():
    st.markdown("<div class='main-title'>📄 Ask Pradeep's Resume</div>", unsafe_allow_html=True)
    st.markdown("<div class='subheader'>Powered by GPT-4 + FAISS | Ask anything about Pradeep's experience, skills, or projects</div>", unsafe_allow_html=True)

    chain = getchain()
    query = st.text_input("🔍 Type your question here:")

    if query:
        with st.spinner("🧠 Thinking..."):
            response = chain.invoke(query)
            result = response.get("result", "No response")
            st.write(result)
            #st.markdown(f"<div class='response-box'>{result}</div>", unsafe_allow_html=True)


# Run app
if __name__ == '__main__':
    main()
