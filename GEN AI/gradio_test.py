import gradio as gr
from langchain_anthropic import ChatAnthropic
import os
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["HF_TOKEN"] = ""

def format_prompt(variable):
    return prompt.format(**variable)

llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        temperature=0.7,
        max_tokens=200
    )

prompt=ChatPromptTemplate.from_messages([
        ("system", """Answer the following question .

            Question:
            {question}"""
        )
])

chain=(
    RunnableLambda(format_prompt)
    | llm
    | StrOutputParser()
)


def process_prompt(prompt):
    response=chain.invoke({"question":prompt})
    return response

chat_application = gr.Interface(
    fn=process_prompt,
    inputs=gr.Textbox(label="Input", lines=2, placeholder="Type your question here..."),
    outputs=gr.Textbox(label="Output"),
)

chat_application.launch()