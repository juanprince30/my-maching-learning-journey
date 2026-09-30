from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence, RunnableLambda
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_classic import chains
from langchain_anthropic import ChatAnthropic

import os

os.environ["ANTHROPIC_API_KEY"] = ""

role = """
    Dungeon & Dragons game master
"""

tone = "engaging and immersive"

template = """
    You are an expert {role}. I have this question {question}. I would like our conversation to be {tone}.
    
    Answer:
    
"""


llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        temperature=0.7,
        max_tokens=1024
    )

    

prompt=PromptTemplate.from_template(template)

def format_prompt(variables):
    return prompt.format(**variables)

chain= (
    RunnableLambda(format_prompt) |
    llm |
    StrOutputParser()
)

# Create an interactive chat loop
while True:
    query = input("Question: ")
    
    if query.lower() in ["quit", "exit", "bye"]:
        print("Answer: Goodbye!")
        break
        
    response = chain.invoke({"role": role, "question": query, "tone": tone})
    print("Answer: ", response)