from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence, RunnableLambda
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_classic import chains
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field
from langchain_pymupdf4llm.pymupdf4llm_loader import PyMuPDF4LLMLoader
from langchain_core.tools import Tool
from langchain.tools import tool
from langchain_experimental.utilities import PythonREPL
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma


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

# class advice(BaseModel):
#     setup: str = Field(description="question to set up the advice")
#     reply: str = Field(description="answer to resolve the question")

# query="what advise can you give to someone who wants to apply for the eiffel france scholarship meanwhile working as a junior developper?"

# output_parser=JsonOutputParser(pydantic_object=advice)
# format_instructions = output_parser.get_format_instructions()

# prompt=ChatPromptTemplate.from_messages([
#     ("system", "Tu es un senior developpeur qui aime aider les jeune. Use this format to return your answer {format}"),
#     ("user", "Pouvez vous m'aider avec cette inquietude svp: {message}")
# ])

# def format_prompt(variable):
#     return prompt.format(**variable)

# chain=(
#     RunnableLambda(format_prompt)
#     | llm
#     | output_parser
# )


# response=chain.invoke({"message": query, "format":format_instructions})


# print(response)

# def format_prompt(variable):
#     return prompt.format(**variable)

# class movie(BaseModel):
#     title: str=Field(decription="This is the title of the movie")
#     director: str=Field(description="This is the name of the director of the movie")
#     year: int =Field(description="This is the year in numerical of the release of that movie")
#     genre: str =Field(description="This is the genre of the movie")

# json_parser=JsonOutputParser(pydantic_object=movie)

# format_instructions = """RESPONSE FORMAT: Return ONLY a single JSON object—no markdown, no examples, no extra keys.  It must look exactly like:
# {
#   "title": "movie title",
#   "director": "director name",
#   "year": 2000,
#   "genre": "movie genre"
# }

# IMPORTANT: Your response must be *only* that JSON.  Do NOT include any illustrative or example JSON."""


# prompt=ChatPromptTemplate.from_messages([
#         ("system", """You are a JSON-only assistant.

#             Task: Generate info about the movie "{movie_name}" in JSON format.

#             {format_instructions}
#             """
#         )
# ])


# chain=(
#     RunnableLambda(format_prompt)
#     | llm
#     | json_parser
# )

# result=chain.invoke({"movie_name": "Lupin", "format_instructions": format_instructions})
# print(result)

# print("Parsed result:")
# print(f"Title: {result['title']}")
# print(f"Director: {result['director']}")
# print(f"Year: {result['year']}")
# print(f"Genre: {result['genre']}")


#------Modern RAG 


loader = PyMuPDF4LLMLoader("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/96-FDF8f7coh0ooim7NyEQ/langchain-paper.pdf")
document = loader.load()

textsplitter=CharacterTextSplitter(chunk_size=200, separator="\n", chunk_overlap=20)
chunks=textsplitter.split_documents(document)

docsearch = Chroma.from_documents(chunks, embeddings)
query = " what are the different type of message we can use with langchain?"

#with retriever
retriever=docsearch.as_retriever()
docs=retriever.invoke(query)


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

response=chain.invoke({"context": format_docs(docs) , "question":query})
print(response)