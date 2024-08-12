import os
import logging
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, trim_messages
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

# Set up logging
logging.basicConfig(filename='chatbot_errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Initialize the model
try:
    model = ChatOpenAI(model="gpt-3.5-turbo")
except Exception as e:
    logging.error(f"Failed to initialize ChatOpenAI model: {str(e)}")
    print("An error occurred while initializing the chatbot. Please check the log file.")
    exit(1)

# Create a store for chat histories
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer all questions to the best of your ability."),
    MessagesPlaceholder(variable_name="messages"),
])

# Create the message trimmer
trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# Create the chain
chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)

# Set up the chain with message history
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

def main():
    session_id = "user_session"
    config = {"configurable": {"session_id": session_id}}

    print("Welcome to the chatbot! Type 'quit' to exit.")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        print("Bot: ", end="", flush=True)
        
        try:
            for chunk in with_message_history.stream(
                {
                    "messages": [HumanMessage(content=user_input)],
                    "language": "English",
                },
                config=config
            ):
                print(chunk.content, end="", flush=True)
            print()  # New line after the complete response
        except Exception as e:
            logging.error(f"Error during chat interaction: {str(e)}")
            print("\nI'm sorry, but I encountered an error. Please try again.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Unhandled exception in main: {str(e)}")
        print("An unexpected error occurred. The chatbot will now exit.")
