import os
import chainlit as cl
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.base.llms.types import ChatMessage
from llama_index.llms.groq import Groq
from qdrant_client import QdrantClient
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
 
# Set Hugging Face API token
HUGGINGFACE_API_TOKEN = "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"
os.environ["HUGGINGFACE_API_TOKEN"] = os.getenv("HUGGINGFACE_API_TOKEN")
 
# Set Groq API token and URL
os.environ['GROQ_TOKEN'] = os.getenv("GROQ_TOKEN")
 
# Setup LLM 
llm = Groq(
    model="llama3-8b-8192",
    api_key=os.environ['GROQ_TOKEN'],
    temperature=0.7,
    api_url="https://api.groq.com/openai/v1"
)
 
# Setup embedding model
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.embed_model = embed_model
Settings.llm = llm
Settings.streaming = True
Settings.num_output = 512
Settings.context_window = 3900
 
# Setup Qdrant vector store
qdrant_client = QdrantClient(
    url="https://c4d519e8-841b-491e-8ffd-a70f8ab07f22.us-east-1-0.aws.cloud.qdrant.io",
    api_key=os.getenv("QDRANT_API_KEY")
)
 
vector_store = QdrantVectorStore(client=qdrant_client, collection_name="franklin_bot")
index = VectorStoreIndex.from_vector_store(vector_store)
 
# Create the chat engine 
chat_engine = index.as_chat_engine(chat_mode="context", llm=llm, verbose=True)
 
 
# Chainlit handlers
@cl.on_chat_start
async def start():
    cl.user_session.set("chat_engine", chat_engine)
    cl.user_session.set("conversation_history", [ChatMessage(role="system", content="You are an assistant helping students with university information.")])
 
@cl.on_message
async def main(message: cl.Message):
    chat_engine = cl.user_session.get("chat_engine")
    conversation_history = cl.user_session.get("conversation_history")
    msg = ChatMessage(role="user", content=message.content)
    conversation_history.append(msg)
 
    response = chat_engine.stream_chat(chat_history=conversation_history, message=message.content)
    msg_out = cl.Message(content="", author="assistant")
 
    for token in response.response_gen:
        await msg_out.stream_token(token)
    await msg_out.send()
 
    conversation_history.append(ChatMessage(role="assistant", content=response.response))
    cl.user_session.set("conversation_history", conversation_history)
