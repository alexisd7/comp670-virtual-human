import os
import chainlit as cl

from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient

# Set HuggingFace token
HUGGINGFACE_API_TOKEN = "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"
os.environ["HUGGINGFACE_API_TOKEN"] = HUGGINGFACE_API_TOKEN

# Qdrant client
qdrant_client = QdrantClient(
    url="https://c4d519e8-841b-491e-8ffd-a70f8ab07f22.us-east-1-0.aws.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.sSiOCw3TUcPi_DmrmRPdruESddRctx1kYBVk8E3voGI"
)

# Qdrant vector store
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="franklin_bot"
)

# Embedding model
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.embed_model = embed_model

# Load the index
index = VectorStoreIndex.from_vector_store(vector_store)

# Create a chat engine
chat_engine = index.as_chat_engine(chat_mode="context",llm=None)

# Chainlit hooksø
@cl.on_chat_start
async def start():
    cl.user_session.set("chat_engine", chat_engine)

@cl.on_message
async def main(message: cl.Message):
    chat_engine = cl.user_session.get("chat_engine")
    response = chat_engine.chat(message.content)
    await cl.Message(content=response.response).send()
