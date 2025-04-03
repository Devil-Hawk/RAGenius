import streamlit as st
import openai
from openai import OpenAI
import faiss
import numpy as np
import io
import toml
from PyPDF2 import PdfReader
import os

# -------------------------------
# Configuration & Secrets
# -------------------------------

st.set_page_config(page_title="PDF IQ Chat", layout="wide")

# Load the API key from Streamlit secrets and initialize OpenAI client
try:
    api_key = st.secrets["openai"]["api_key"]
    
    # Initialize OpenAI client directly with the key
    client = OpenAI(api_key=api_key)

except KeyError:
    st.error("OpenAI API key not found in Streamlit secrets. Please add it via the Streamlit Cloud interface.")
    client = None
except Exception as e:
    st.error(f"Error initializing OpenAI client from secrets: {str(e)}")
    client = None

# -------------------------------
# Session State Initialization
# -------------------------------

if "documents" not in st.session_state:
    st.session_state.documents = []  # List of tuples: (text, embedding)
if "faiss_index" not in st.session_state:
    embedding_dim = 1536  # Dimension for text-embedding-ada-002
    st.session_state.faiss_index = faiss.IndexFlatL2(embedding_dim)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# -------------------------------
# Helper Functions
# -------------------------------

def get_embedding(text: str) -> np.ndarray:
    """Call OpenAI's Embedding API to get an embedding for the provided text."""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    embedding = response.data[0].embedding
    return np.array(embedding, dtype=np.float32)

def extract_text(file) -> str:
    """Extract text from a PDF or text file."""
    content = file.read()
    if file.name.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(content))
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    else:
        text = content.decode("utf-8")
    return text

# -------------------------------
# UI: File Upload Section
# -------------------------------

st.title("PDF IQ Chat")
st.markdown("### Upload your Documents")
st.markdown("*(Upload PDFs or text files. The content will be indexed for chat.)*")

uploaded_files = st.file_uploader("Select files", type=["pdf", "txt"], accept_multiple_files=True)

if st.button("Upload Files"):
    if not uploaded_files:
        st.error("No files selected for upload.")
    else:
        for file in uploaded_files:
            text = extract_text(file)
            embedding = get_embedding(text)
            st.session_state.documents.append((text, embedding))
            # Add the embedding to the FAISS index
            st.session_state.faiss_index.add(np.expand_dims(embedding, axis=0))
        st.success(f"Uploaded {len(uploaded_files)} file(s) successfully.")

st.markdown("---")
st.markdown("### Chat with Your Documents")

# -------------------------------
# Chat: Send Message with Limit
# -------------------------------

def send_message():
    # Count the number of user messages in chat history
    user_message_count = sum(1 for msg in st.session_state.chat_history if msg["role"] == "user")
    if user_message_count >= 20:
        st.error("You have reached the maximum of 20 messages. Please restart your session to talk again.")
        return

    if st.session_state.input_text.strip() == "":
        st.error("Please enter a valid question.")
        return

    # Append the user message to the chat history
    user_message = {"role": "user", "text": st.session_state.input_text}
    st.session_state.chat_history.append(user_message)

    # Retrieve relevant document context using FAISS
    query_embedding = get_embedding(st.session_state.input_text)
    k = 3  # Retrieve top 3 documents
    distances, indices = st.session_state.faiss_index.search(np.expand_dims(query_embedding, axis=0), k)
    context = ""
    for idx in indices[0]:
        if idx < len(st.session_state.documents):
            context += st.session_state.documents[idx][0] + "\n"

    # Construct the prompt
    prompt = (
        "Answer the question based on the context below.\n\n"
        f"Context:\n{context}\n"
        f"Question: {st.session_state.input_text}\nAnswer:"
    )

    # Clear the input field
    st.session_state.input_text = ""
    
    try:
        # Call OpenAI's ChatCompletion API
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
        st.session_state.chat_history.append({"role": "assistant", "text": answer})
    except Exception as e:
        st.error("Error: " + str(e))

# Text input widget for user messages
st.text_input("Type your question:", key="input_text", on_change=send_message)

# -------------------------------
# Display Chat History
# -------------------------------

st.markdown("#### Conversation")
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(
            f"<div style='text-align: right; color: lightblue;'><strong>You:</strong> {msg['text']}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div style='text-align: left; color: lightgreen;'><strong>Assistant:</strong> {msg['text']}</div>",
            unsafe_allow_html=True
        )
    