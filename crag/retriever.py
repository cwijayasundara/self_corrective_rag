import os
from dotenv import load_dotenv
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub

load_dotenv()

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'

prompt = hub.pull("rlm/rag-prompt")

persistent_dir = "db"

# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)

vectorstore = Chroma(persist_directory=persistent_dir,
                     embedding_function=OpenAIEmbeddings())


def get_retriever():
    return vectorstore.as_retriever()

# Testing
# query = "Whats the additional payment for buildings or contents ?"
# response = get_retriever().invoke(query)
# print(response)
