import os
import io
import time
import toml  # pip install toml
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import openai
from openai import OpenAI  # Add this import
import faiss
import numpy as np
import uvicorn
from PyPDF2 import PdfReader
from logging_config import setup_logging

# Initialize logger
logger = setup_logging()

app = FastAPI()

# Enable CORS for frontend access.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load OpenAI API key exclusively from the secrets file in rag-streamlit/.streamlit
# Directory structure:
# A:\RAG\
#   ├── rag-backend\   (this file is here)
#   ├── rag-frontend\
#   └── rag-streamlit\
#         ├── app.py
#         └── .streamlit\
#               └── secrets.toml
secrets_path = os.path.join(os.path.dirname(__file__), "..", "rag-streamlit", ".streamlit", "secrets.toml")
secrets_data = toml.load(secrets_path)
api_key = secrets_data["openai"]["api_key"]

# Initialize OpenAI client directly with the key
try:
    client = OpenAI(api_key=api_key)
except Exception as e:
    logger.error(f"Error initializing OpenAI client: {str(e)}")
    # Optionally raise an error or exit if client initialization fails
    raise HTTPException(status_code=500, detail="Failed to initialize OpenAI client.")

# Global in-memory storage for documents and FAISS index
documents = []  # List of tuples: (text, embedding)
embedding_dim = 1536  # For text-embedding-ada-002
index = faiss.IndexFlatL2(embedding_dim)

def get_embedding(text: str) -> np.ndarray:
    """Generate an embedding for the given text using OpenAI's API."""
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        embedding = response.data[0].embedding
        return np.array(embedding, dtype=np.float32)
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        try:
            logger.info(f"Processing file upload: {file.filename}")
            content = await file.read()
            # Handle PDFs vs. text files
            if file.filename.lower().endswith(".pdf"):
                reader = PdfReader(io.BytesIO(content))
                text = ""
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            else:
                text = content.decode("utf-8")
            
            # Generate embedding and store document
            embedding = get_embedding(text)
            documents.append((text, embedding))
            index.add(np.expand_dims(embedding, axis=0))
            results.append({"filename": file.filename, "status": "uploaded"})
            logger.info(f"Successfully processed file: {file.filename}")
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {str(e)}")
            results.append({"filename": file.filename, "status": "error", "error": str(e)})
    return {"files": results}

# Rate limiting configuration
RATE_LIMIT_SECONDS = 2  # Minimum seconds between requests per IP
MAX_CONVERSATIONS_PER_IP = 20  # Maximum total conversations allowed per IP
ip_last_request = {}       # To store the timestamp of the last request per IP
ip_conversation_count = {} # To count total conversations per IP

@app.post("/ask")
async def ask_question(request: Request, question: str = Form(...)):
    client_ip = request.client.host
    now = time.time()
    logger.info(f"Received question from IP {client_ip}: {question}")

    # Enforce a minimum time gap between requests from the same IP.
    last_request = ip_last_request.get(client_ip, 0)
    if now - last_request < RATE_LIMIT_SECONDS:
        logger.warning(f"Rate limit exceeded for IP {client_ip}")
        raise HTTPException(status_code=429, detail="Too many requests, please slow down.")
    ip_last_request[client_ip] = now

    # Increment conversation count for this IP and check against limit.
    ip_conversation_count[client_ip] = ip_conversation_count.get(client_ip, 0) + 1
    if ip_conversation_count[client_ip] > MAX_CONVERSATIONS_PER_IP:
        logger.warning(f"Conversation limit reached for IP {client_ip}")
        raise HTTPException(status_code=429, detail="Conversation limit reached for this IP.")

    if index.ntotal == 0:
        logger.error("No documents uploaded when attempting to ask a question")
        return JSONResponse(status_code=400, content={"error": "No documents uploaded."})
    
    # Generate embedding for the question and search for the most relevant documents.
    query_embedding = get_embedding(question)
    k = 3  # Number of documents to retrieve
    distances, indices = index.search(np.expand_dims(query_embedding, axis=0), k)
    context = ""
    for idx in indices[0]:
        if idx < len(documents):
            context += documents[idx][0] + "\n"

    prompt = (
        "Answer the question based on the context below.\n\n"
        f"Context:\n{context}\n"
        f"Question: {question}\nAnswer:"
    )

    # Use OpenAI ChatCompletion API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0.2
    )
    answer = response.choices[0].message.content.strip()
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
