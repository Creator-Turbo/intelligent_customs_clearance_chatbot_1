from flask import Flask, render_template, request, jsonify
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
# from huggingface_hub import InferenceClient
from langchain_groq import ChatGroq
from langchain_core.chat_history import InMemoryChatMessageHistory

from dotenv import load_dotenv
from src.prompt import *
import os 



app = Flask(__name__)
load_dotenv()
# Load environment variables from .env file


# Get API keys from environment variables
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')


# Validate if the keys are loaded correctly
if not PINECONE_API_KEY:
    raise ValueError("Error: PINECONE_API_KEY is not set. Check your .env file.")
if not HUGGINGFACEHUB_API_TOKEN:
    raise ValueError("Error: HUGGINGFACEHUB_API_TOKEN is not set. Check your .env file.")

# Store them explicitly in environment variables
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN

print("API keys loaded successfully!")

## using the hugging_face llms model 
embeddings = download_hugging_face_embeddings()


index_name = "customs-clearance-chatbot"

# Embed each chunk and upsert the embeddings into your Pinecone index 

docsearch = PineconeVectorStore.from_existing_index(
    index_name = index_name,
    embedding = embeddings
)


retriever = docsearch.as_retriever(search_type = "similarity",search_kwargs ={"k":3})


# llm = ChatGroq(
#     model="llama-3.3-70b-versatile",   # Other options: llama3-8b-8192, mixtral-8x7b
#     temperature=0.7
# )


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    max_tokens=512,
    # reasoning_format="parsed",
    max_retries=3,
)



prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),   # Add this line
        ("human", "{input}"),
    ]
)

# memory store for each session
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]



question_answer_chain = create_stuff_documents_chain(llm, prompt)

rag_chain = create_retrieval_chain(retriever, question_answer_chain)



# Wrap with memory
rag_chain_with_memory = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",   # what user says
    history_messages_key="chat_history",  # stored chat
    output_messages_key="answer"  # model’s reply
)




@app.route("/")
def index():
    return render_template('index.html') 


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User:", msg)

    # Session ID can be static or per-user
    session_id = "default_user"

    response = rag_chain_with_memory.invoke(
        {"input": msg},
        config={"configurable": {"session_id": session_id}}
    )

    # print("Response:", response["answer"])
    # print("Chat History:", get_session_history(session_id).messages)
    return str(response["answer"])



if __name__ == "__main__":
     app.run(host="0.0.0.0", port=5000
             , debug=True)



