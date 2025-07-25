import os
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import streamlit as st

# Load environment variables
load_dotenv()
parser = StrOutputParser()

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
template = """
You are a helpful assistant that answers questions based on retrieved documents.

Use the conversation history below to answer the question.

If the user asks about previous questions they've asked, list all the human questions from the conversation history.

Context:
{context}

Conversation History:
{chat_history}

Question: {question}

Answer:
"""
prompt = PromptTemplate(template=template,
                        input_variables=['chat_history', 'question'])

# Chain creation function
def getchain():
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.load_local('gunda_vector_store', embeddings=embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={'k': 3})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key),
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt})
    return qa_chain

# Main app
def main():
    if "chain" not in st.session_state:
        st.session_state.chain = getchain()
    st.markdown("<div class='main-title'>📄 Ask Pradeep's Resume</div>", unsafe_allow_html=True)
    st.markdown("<div class='subheader'>Powered by GPT-4 + FAISS | Ask anything about Pradeep's experience, skills, or projects</div>", unsafe_allow_html=True)

    query = st.text_input("🔍 Type your question here:")

    if query:
        with st.spinner("🧠 Thinking..."):
            response = st.session_state.chain.invoke({"question": query})
            result = response.get("answer", "No response")
            st.write(result)
            #st.write(st.session_state.chain.memory.chat_memory.messages)
            #nq_prompt=PromptTemplate(template="""Based on the below chat history, suggest a good next question
            #chat_history:{chat_history}""",input_variables=['chat_history'])
            model = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key)
            #nq_chain = nq_prompt | model | parser
            #nq=nq_chain.invoke({"chat_history": st.session_state.chain.memory.chat_memory.messages})
            #st.write(nq)


            #st.markdown(f"<div class='response-box'>{result}</div>", unsafe_allow_html=True)


# Run app
if __name__ == '__main__':
    main()
