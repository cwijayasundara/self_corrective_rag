import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

load_dotenv()

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'

embedding = OpenAIEmbeddings()

file_path = "../docs/QM485-1122_Business_Pack_Insurance_Policy.pdf"

loader = PyPDFLoader("../docs/QM485-1122_Business_Pack_Insurance_Policy.pdf")

docs = loader.load()

# Split

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1024,
    chunk_overlap=50)

# Make splits
splits = text_splitter.split_documents(docs)

# Index

vectorstore = Chroma.from_documents(documents=splits,
                                    embedding=embedding,
                                    persist_directory="db")

# for testing
# retriever = vectorstore.as_retriever()
# query = "Whats the additional payment for buildings or contents ?"
# response = retriever.invoke(query)
# print(response)
