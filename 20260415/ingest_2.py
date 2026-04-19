import os
import json
from pathlib import Path
from typing import Any

from langchain.docstore.document import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_upstage import UpstageEmbeddings

# ==============================
# 설정
# ==============================
JSON_PATH = Path("./document_list.json")
PERSIST_DIR = "./chroma"
COLLECTION_NAME = "chroma-PA_tax_v3"

STRICT_SOURCE_VALIDATION = False
ALLOW_EMPTY_SOURCE_FOR_TYPES = {"guide", "common"}

# ==============================
# JSON 로드
# ==============================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
)

loader = Docx2txtLoader('./randover.docx')
document_list = loader.load_and_split(text_splitter=text_splitter)

embedding = UpstageEmbeddings(model="solar-embedding-1-large", api_key="up_EAlYOcqCPb9gq6b64Ecq6lz1sihol")

database = Chroma.from_documents(documents=document_list, embedding=embedding, collection_name=COLLECTION_NAME, persist_directory=PERSIST_DIR)