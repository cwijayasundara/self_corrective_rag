from dotenv import load_dotenv
import os

from langchain_community.vectorstores.chroma import Chroma

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'

prompt = hub.pull("rlm/rag-prompt")

persistent_dir = "db"

# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)

vectorstore = Chroma(persist_directory=persistent_dir,
                     embedding_function=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Chain
rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

# Question
response = rag_chain.invoke("Whats the additional payment for buildings or contents ?")
print(response)
