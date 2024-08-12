import getpass
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, trim_messages

from operator import itemgetter

from langchain_core.runnables import RunnablePassthrough

load_dotenv()
# 
model = ChatOpenAI(model="gpt-3.5-turbo")
# 
# response = model.invoke([HumanMessage(content="Hi! I'm Bob")])
# 
# print(response)
# print(type(response))
# 
# r2 = model.invoke(
    # [
        # HumanMessage(content="Hi! I'm Bob"),
        # AIMessage(content="Hello Bob! How can I assist you today?"),
        # HumanMessage(content="What's my name?"),
    # ]
# )
# 
# print(r2)
# print(type(r2))

# # message histories
# 
store = { }
# 
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
# 
# 
# with_message_history = RunnableWithMessageHistory(model, get_session_history)
# 
# config = {"configurable": {"session_id": "abc2"}}
# 
# response = with_message_history.invoke(
    # [HumanMessage(content="Hi! I'm Bob")],
    # config=config,
# )
# 
# print(response.content)
# 
# response = with_message_history.invoke(
    # [HumanMessage(content="What's my name?")],
    # config=config,
# )
# 
# print(response.content)
# 
# config = {"configurable": {"session_id": "abc3"}}
# 
# response = with_message_history.invoke(
    # [HumanMessage(content="What's my name?")],
    # config=config,
# )
# 
# print(response.content)
# 
# config = {"configurable": {"session_id": "abc2"}}
# 
# response = with_message_history.invoke(
    # [HumanMessage(content="What's my name?")],
    # config=config,
# )
# 
# print(response.content)

# prompt templates

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model

response = chain.invoke({"messages": [HumanMessage(content="hi! I'm bob")]})

print(response.content)

with_message_history = RunnableWithMessageHistory(chain, get_session_history)

config = {"configurable": {"session_id": "abc5"}}

response = with_message_history.invoke(
    [HumanMessage(content="Hi! I'm Jim")],
    config=config,
)

print(response.content)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model

response = chain.invoke(
    {"messages": [HumanMessage(content="hi! I'm bob")], "language": "Spanish"}
)

print(response.content)

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

config = {"configurable": {"session_id": "abc11"}}

response = with_message_history.invoke(
    {"messages": [HumanMessage(content="hi! I'm todd")], "language": "Spanish"},
    config=config,
)

trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

print(trimmer.invoke(messages))

chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)

response = chain.invoke(
    {
        "messages": messages + [HumanMessage(content="what's my name?")],
        "language": "English",
    }
)

print(response)

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

config = {"configurable": {"session_id": "abc20"}}

response = with_message_history.invoke(
    {
        "messages": messages + [HumanMessage(content="whats my name?")],
        "language": "English",
    },
    config=config,
)

print(response)

config = {"configurable": {"session_id": "abc15"}}
for r in with_message_history.stream(
    {
        "messages": [HumanMessage(content="hi! I'm todd. tell me a joke")],
        "language": "English",
    },
    config=config,
):
    print(r.content, end="|")
