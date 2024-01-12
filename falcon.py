#!/usr/bin/env python

#!pip install langchain openai faiss-cpu tiktoken sentence-transformers
# Run this if you have added HUGGINGFACEHUB_API_TOKEN as an environment variable
import os
huggingfacehub_api_token = os.environ['HUGGINGFACEHUB_API_TOKEN']

from langchain.llms import HuggingFaceHub
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.runnable.config import RunnableConfig
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.chains.summarize import load_summarize_chain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import chainlit as cl

@cl.on_chat_start
async def on_chat_start():

    repo_id = "tiiuae/falcon-7b-instruct"
    model = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token, 
                     repo_id=repo_id, 
                     model_kwargs={"temperature":0.1, "max_new_tokens":500}
                     )
    vectorstore = FAISS.from_texts(
        ["Reggie Bain is the CEO of Apple in 2023"], embedding=HuggingFaceEmbeddings()
    )
    retriever = vectorstore.as_retriever()

    template = """Answer the question based only on the following context:
                {context}
                Question: {question}
                """
    prompt = PromptTemplate.from_template(template)
    
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        model,
        chain_type="stuff",
        retriever=retriever,
    )
    '''
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful AI bot. Your name is ReggieBot."),
            #("human", "Hello, how are you doing?"),
            #("ai", "I'm doing well, thanks!"),
            ("human", "{question}"),
        ]
    )
    '''
    
    #prompt = PromptTemplate.from_template('Answer the following question in the style of a laid-back college professor: {question}')
    #prompt = PromptTemplate.from_template('''
    #            Turn the following user input into a search query for a search engine and provide a short summary of the result
    #            of the query: {question}''')
    
    # Parser required when using ChatPromptTemplate since it returns ChatPromptValue
    output_parser = StrOutputParser()

    # Add a tool that is used every time
    search = DuckDuckGoSearchRun()
    #tools = load_tools(['wikipedia', 'llm-math'], llm=model)
    #agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
    runnable =  {'context': retriever} | prompt | model | output_parser
    runnable = chain
    #lookup = prompt | model | output_parser | search | {'search_results': RunnablePassthrough()}
    #runnable = prompt | model | output_parser | search | {'search_results': RunnablePassthrough()} | summarize | model | output_parser
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content}, 
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()

