from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import  RunnableLambda
from langchain_anthropic import ChatAnthropic
from langchain_pymupdf4llm.pymupdf4llm_loader import PyMuPDF4LLMLoader
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
import gradio as gr


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["HF_TOKEN"] = ""

llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        temperature=0.7,
        max_tokens=200
    )


loader = PyMuPDF4LLMLoader("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/96-FDF8f7coh0ooim7NyEQ/langchain-paper.pdf")
document = loader.load()

textsplitter=CharacterTextSplitter(chunk_size=200, separator="\n", chunk_overlap=20)
chunks=textsplitter.split_documents(document)

docsearch = Chroma.from_documents(chunks, embeddings)

retriever=docsearch.as_retriever()

def format_prompt(variable):
    return prompt.format(**variable)

def format_docs(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )

prompt=ChatPromptTemplate.from_messages([
        ("system", """Answer the question using only the following context.

            Context:
            {context}

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
    docs=retriever.invoke(prompt)
    response=chain.invoke({"context": format_docs(docs) , "question":prompt})
    return response

chat_application = gr.Interface(
    fn=process_prompt,
    inputs=gr.Textbox(label="Input", lines=2, placeholder="Type your question here..."),
    outputs=gr.Textbox(label="Output"),
)

chat_application.launch()