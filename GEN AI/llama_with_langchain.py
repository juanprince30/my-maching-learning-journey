from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import  RunnableLambda
from langchain_anthropic import ChatAnthropic
from langchain_pymupdf4llm.pymupdf4llm_loader import PyMuPDF4LLMLoader
import os
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
import gradio as gr
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, Settings

os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["HF_TOKEN"] = ""

docs=SimpleDirectoryReader("C:/Users/hp/Desktop/COURSERA AI LEARNING/GEN AI get started/data").load_data()

Settings.embed_model= HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

Settings.llm= Anthropic(
        model="claude-haiku-4-5-20251001",
        temperature=0.7,
        max_tokens=200
    )

index = VectorStoreIndex.from_documents(docs)

query_engine = index.as_query_engine()

response = query_engine.query(
    "Quelles sont les derniers projets ?"
)

print(response)