import os
import io
import toml  # Make sure to install this: pip install toml
from typing import List
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import openai
import faiss
import numpy as np
import uvicorn
from PyPDF2 import PdfReader

app = FastAPI()

# Enable CORS for frontend access.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load OpenAI API key exclusively from rag-streamlit/.streamlit/secrets.toml
# Assuming your directory structure is:
# A:\RAG\
#   ├── rag-backend\   (this file is here)
#   ├── rag-frontend\
#   └── rag-streamlit\
#         ├── app.py
#         └── .streamlit\
#               └── secrets.toml
secrets_path = os.path.join(os.path.dirname(__file__), "..", "rag-streamlit", ".streamlit", "secrets.toml")
secrets_data = toml.load(secrets_path)
openai.api_key = secrets_data["openai"]["api_key"]

documents = []
embedding_dim = 1536
index = faiss.IndexFlatL2(embedding_dim)

def get_embedding(text: str) -> np.ndarray:
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    embedding = response["data"][0]["embedding"]
    return np.array(embedding, dtype=np.float32)

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
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

        # Embed & store document
        embedding = get_embedding(text)
        documents.append((text, embedding))
        index.add(np.expand_dims(embedding, axis=0))
        results.append({"filename": file.filename, "status": "uploaded"})
    return {"files": results}

@app.post("/ask")
async def ask_question(question: str = Form(...)):
    if index.ntotal == 0:
        return JSONResponse(status_code=400, content={"error": "No documents uploaded."})
    query_embedding = get_embedding(question)
    k = 3  # number of documents to retrieve
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
    response = openai.ChatCompletion.create(
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
