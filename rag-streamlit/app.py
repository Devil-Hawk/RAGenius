import streamlit as st
import openai
import faiss
import numpy as np
import io
from PyPDF2 import PdfReader

# -------------------------------
# Configuration and Initialization
# -------------------------------

# Set your OpenAI API key; you can also store this in st.secrets
# Example: st.secrets["OPENAI_API_KEY"]
openai.api_key = st.secrets["openai"]["api_key"]

# Set page configuration (you can choose dark theme via your Streamlit settings)
st.set_page_config(page_title="PDF IQ Chat", layout="wide")

# Initialize session state for documents, FAISS index, and chat history
if "documents" not in st.session_state:
    st.session_state.documents = []  # Each entry: (text, embedding)
if "faiss_index" not in st.session_state:
    embedding_dim = 1536  # For text-embedding-ada-002
    st.session_state.faiss_index = faiss.IndexFlatL2(embedding_dim)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------
# Helper Functions
# -------------------------------

def get_embedding(text: str) -> np.ndarray:
    """Call OpenAI's embedding API for a given text."""
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    embedding = response["data"][0]["embedding"]
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
# UI Components
# -------------------------------

st.title("PDF IQ Chat")
st.markdown("### Upload your PDF or Text Documents")
st.markdown("*(This system ingests documents and lets you chat with their content.)*")

# File uploader widget (supports multiple files)
uploaded_files = st.file_uploader("Choose files", type=["pdf", "txt"], accept_multiple_files=True)

if st.button("Upload Files"):
    if not uploaded_files:
        st.error("No files selected. Please choose at least one file.")
    else:
        for file in uploaded_files:
            text = extract_text(file)
            embedding = get_embedding(text)
            st.session_state.documents.append((text, embedding))
            # Add embedding to FAISS index
            vector = np.expand_dims(embedding, axis=0)
            st.session_state.faiss_index.add(vector)
        st.success(f"Uploaded {len(uploaded_files)} file(s) successfully.")

st.markdown("---")
st.markdown("### Chat with Your Documents")

# Text input for questions
question = st.text_input("Type your question here:", key="question_input")

if st.button("Send Question"):
    if not question.strip():
        st.error("Please enter a valid question.")
    elif st.session_state.faiss_index.ntotal == 0:
        st.error("No documents available. Please upload files first.")
    else:
        # Append user's question to chat history
        st.session_state.chat_history.append({"role": "user", "text": question})
        query_embedding = get_embedding(question)
        k = 3  # number of top documents to retrieve
        distances, indices = st.session_state.faiss_index.search(np.expand_dims(query_embedding, axis=0), k)
        context = ""
        for idx in indices[0]:
            if idx < len(st.session_state.documents):
                context += st.session_state.documents[idx][0] + "\n"
        prompt = (
            "Answer the question based on the context below.\n\n"
            f"Context:\n{context}\n"
            f"Question: {question}\nAnswer:"
        )
        # Call OpenAI ChatCompletion endpoint with the constructed prompt
        try:
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
            st.session_state.chat_history.append({"role": "assistant", "text": answer})
        except Exception as e:
            st.error("Error: " + str(e))

# -------------------------------
# Display Chat History
# -------------------------------
st.markdown("### Conversation")
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"<div style='text-align: right; color: lightblue;'><strong>You:</strong> {msg['text']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align: left; color: lightgreen;'><strong>Assistant:</strong> {msg['text']}</div>", unsafe_allow_html=True)
