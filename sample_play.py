from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from dotenv import  load_dotenv
from langchain_community.tools import tool
from langchain.agents import AgentType
AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION

from nltk import toolbox

from ingest import openai_api_key

load_dotenv()
llm = ChatOpenAI(openai_api_key=openai_api_key)

@tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b
@tool
def multiply(a: int, b: int)-> int:
    """Multiply two numbers"""
    return a * b






llm_with_tools = llm.bind_tools([add, multiply])
print(llm_with_tools.invoke("what is 3 plus 5"))