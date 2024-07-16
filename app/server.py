from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from decouple import config
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "pdf-index"

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

embeddings=OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vectorstore = PineconeVectorStore(
    index_name=index_name, 
    embedding=embeddings
)
model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
template = """
Answer the question based on the context below. If you can't 
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
parser = StrOutputParser()
chain = (
    {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | model
    | parser
)

# model = ChatOpenAI(openai_api_key=config("OPENAI_API_KEY"))
# prompt = ChatPromptTemplate.from_template("Give me a summary about {topic} in a paragraph or less.")
# chain = prompt | model

# Edit this to add the chain you want to add
# add_routes(app, NotImplemented)
add_routes(app, chain, path="/openai")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
