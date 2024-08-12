import os
import logging
from dotenv import load_dotenv
from flask import Flask, request, Response, stream_with_context
from flask_cors import CORS, cross_origin
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import trim_messages
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough


# Set up logging
logging.basicConfig(filename='chatbot_errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow requests from Svelte dev server

app.config['CORS_HEADERS'] = 'Content-Type'

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

# Create a more customizable prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Answer all questions to the best of your ability "
        "in {language}. Speak at a {speaking_level} level with a {tone} tone. "
        "Try to incorporate these words or phrases if relevant: {specific_words}. "
        "Additional instructions: {additional_instructions}"
    ),
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

@app.route('/chat', methods=['POST', 'OPTIONS'])
@cross_origin()
def chat():
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = app.make_default_options_response()
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Session-ID'
        return response

    data = request.json
    user_input = data.get('message', '')
    settings = data.get('settings', {})
    session_id = request.headers.get('X-Session-ID', 'default_session')

    config = {"configurable": {"session_id": session_id}}

    def generate():
        try:
            for chunk in with_message_history.stream(
                {
                    "messages": [HumanMessage(content=user_input)],
                    **settings
                },
                config=config
            ):
                yield chunk.content
        except Exception as e:
            logging.error(f"Error during chat interaction: {str(e)}")
            yield "I'm sorry, but I encountered an error. Please try again."

    return Response(stream_with_context(generate()), content_type='text/plain')

if __name__ == '__main__':
    app.run(debug=True)
