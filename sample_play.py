from langchain_community.tools import tool

@tool
def multiply(a:int, b:int) ->int:
    """multiply two numbers"""
    return a*b



result = multiply.invoke({'a':10, 'b':20})
print(result)
print(multiply.name)
print(multiply.description)
print(multiply.args)