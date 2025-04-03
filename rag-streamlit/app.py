import streamlit as st

# import openai  # F401 - Keep commented or remove if truly unused
from openai import OpenAI
import faiss
import numpy as np
import io

# import toml  # F401 - Keep commented or remove if truly unused
from PyPDF2 import PdfReader

# import os  # F401 - Keep commented or remove if truly unused

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
    st.error(
        "OpenAI API key not found in Streamlit secrets. Please add it via the Streamlit Cloud interface."
    )
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
# Add a check for consistency, though this shouldn't normally happen
elif not hasattr(st.session_state.faiss_index, "ntotal"):
    st.warning("Re-initializing FAISS index due to unexpected state.")
    embedding_dim = 1536
    st.session_state.faiss_index = faiss.IndexFlatL2(embedding_dim)
    st.session_state.documents = []  # Also clear documents if index is bad
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# Add a flag to disable button during processing
if "processing_upload" not in st.session_state:
    st.session_state.processing_upload = False


# -------------------------------
# Helper Functions
# -------------------------------


def get_embedding(text: str) -> np.ndarray:
    """Call OpenAI's Embedding API to get an embedding for the provided text."""
    response = client.embeddings.create(input=text, model="text-embedding-ada-002")
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


def chunk_text(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 150
) -> list[str]:
    """Splits text into overlapping chunks."""
    if not isinstance(text, str):
        return []
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        # Move start forward, ensuring overlap and preventing infinite loops
        next_start = start + chunk_size - chunk_overlap
        # If overlap is too large or chunk size too small, move forward by chunk size
        if next_start <= start:
            next_start = start + chunk_size
        start = next_start
        # No need to break explicitly, the while condition handles the end

    return [chunk for chunk in chunks if chunk.strip()]  # Remove empty chunks


def process_uploaded_files(uploaded_files, client):
    """Handles text extraction, chunking, embedding, and indexing for uploaded files."""
    if not uploaded_files:
        st.error("No files selected for upload.")
        return None, None  # Return None if no files

    if client is None:
        st.error("OpenAI client not initialized. Check secrets configuration.")
        return None, None  # Return None if client bad

    new_documents: list[tuple[str, np.ndarray, str]] = []
    embedding_dim = 1536
    current_batch_index = faiss.IndexFlatL2(embedding_dim)
    processing_error = False
    files_processed_count = 0
    chunks_processed_count = 0

    with st.spinner("Extracting text from documents..."):
        all_chunks_with_source: list[tuple[str, str]] = []
        for file in uploaded_files:
            full_text = extract_text(file)
            if full_text:
                file_chunks = chunk_text(full_text)
                for chunk in file_chunks:
                    all_chunks_with_source.append((chunk, file.name))
                files_processed_count += 1
            else:
                st.warning(
                    f"Skipping file {file.name} due to text extraction issues.",
                    icon="⚠️",
                )

        if not all_chunks_with_source:
            st.error("No text could be extracted or chunked from the selected files.")
            return None, None  # Return None if no chunks

    if all_chunks_with_source:
        with st.spinner("Indexing documents..."):
            try:
                for i, (chunk, filename) in enumerate(all_chunks_with_source):
                    embedding = get_embedding(chunk)
                    new_documents.append((chunk, embedding, filename))
                    current_batch_index.add(np.expand_dims(embedding, axis=0))
                    chunks_processed_count += 1

                st.success(
                    f"Successfully processed {files_processed_count} file(s) "
                    f"({chunks_processed_count} sections indexed). Ready for questions!",
                    icon="✅",
                )
                return new_documents, current_batch_index  # Return successful results

            except Exception as e:
                st.error(f"Error during document processing: {str(e)}", icon="🚨")
                return None, None  # Return None on error
    else:
        st.warning("No text could be processed from the uploaded files.", icon="⚠️")
        return None, None  # Return None if no chunks processed


# -------------------------------
# UI: File Upload Section
# -------------------------------

st.title("PDF IQ Chat")
st.markdown("### Upload your Documents")
st.markdown("*(Upload PDFs or text files. The content will be indexed for chat.)*")

uploaded_files = st.file_uploader(
    "Select files", type=["pdf", "txt"], accept_multiple_files=True, key="file_uploader"
)

if st.button(
    "Upload Files",
    key="upload_button",
    disabled=st.session_state.get("processing_upload", False),
):
    st.session_state.processing_upload = True
    new_docs, new_index = process_uploaded_files(uploaded_files, client)
    # Only update state if processing was successful
    if new_docs is not None and new_index is not None:
        st.session_state.documents = new_docs
        st.session_state.faiss_index = new_index
        st.session_state.chat_history = []  # Clear history on new upload
    st.session_state.processing_upload = False
    st.rerun()

st.markdown("---")
st.markdown("### Chat with Your Documents")


# -------------------------------
# Chat: Send Message with Limit
# -------------------------------


def send_message():
    # Add verbose state check at the beginning
    # st.write(f"DEBUG: Start send_message. Index ntotal: {getattr(st.session_state.get('faiss_index'), 'ntotal', 'N/A')}, Num docs: {len(st.session_state.get('documents', []))}")

    # Count the number of user messages in chat history
    user_message_count = sum(
        1 for msg in st.session_state.chat_history if msg["role"] == "user"
    )
    if user_message_count >= 20:
        st.error(
            "You have reached the maximum of 20 messages. Please restart your session to talk again."
        )
        return

    if client is None:
        st.error("OpenAI client not initialized. Check secrets configuration.")
        return

    # Check if index exists and has items *before* trying to use it
    if (
        not hasattr(st.session_state, "faiss_index")
        or st.session_state.faiss_index.ntotal == 0
    ):
        st.error("No documents have been uploaded or indexed yet.")
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "text": "Please upload documents before asking questions.",
            }
        )
        return

    # Capture input from session state (set by form submission)
    current_input = st.session_state.input_text

    # Append the user message to the chat history
    user_message = {"role": "user", "text": current_input}
    st.session_state.chat_history.append(user_message)

    # --- Add Check for Meta Question --- 
    normalized_input = current_input.lower().strip().replace("?", "") # Normalize
    # Check for keywords instead of just startswith
    is_count_query = (
        ("how many" in normalized_input or "count" in normalized_input) and 
        ("document" in normalized_input or "file" in normalized_input or "pdf" in normalized_input)
    )

    if is_count_query:
        if "documents" in st.session_state and st.session_state.documents:
            # Calculate unique filenames
            unique_filenames = set(filename for _, _, filename in st.session_state.documents)
            num_docs = len(unique_filenames)
            answer = f"You have successfully processed {num_docs} unique document(s):\n" + "\n".join(f"- {name}" for name in sorted(list(unique_filenames)))
        else:
            answer = "No documents have been successfully processed yet."
        
        st.session_state.chat_history.append({"role": "assistant", "text": answer})
        st.rerun() # Rerun to update the chat display immediately
        # return # No longer strictly needed due to rerun, but good practice
    # --- End Check for Meta Question ---

    try:
        # Retrieve relevant document context using FAISS
        # st.write(f"DEBUG: Searching index (ntotal={st.session_state.faiss_index.ntotal}) for query: {current_input[:50]}...")
        query_embedding = get_embedding(current_input)
        k = min(
            3, st.session_state.faiss_index.ntotal
        )  # Don't ask for more docs than exist

        distances, indices = st.session_state.faiss_index.search(
            np.expand_dims(query_embedding, axis=0), k
        )
        # st.write(f"DEBUG: FAISS search returned indices: {indices}, distances: {distances}")

        context = ""
        valid_indices_found = 0
        if indices.size > 0:
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(st.session_state.documents):
                    # Retrieve chunk text AND filename
                    chunk_text, _, doc_filename = st.session_state.documents[idx]
                    # Prepend filename to the context chunk
                    context += (
                        f"--- Context from '{doc_filename}': ---\n{chunk_text}\n\n"
                    )
                    valid_indices_found += 1
                else:
                    st.warning(
                        f"Invalid index {idx} found at search result position {i}."
                    )

        if valid_indices_found == 0:
            st.warning(
                "Could not find relevant context in the uploaded documents for your query."
            )
            context = "No relevant context could be retrieved."  # Add fallback context

        # Construct the prompt - Instruct LLM about the source labels
        prompt = (
            f"You are comparing information from different documents based *only* on the context provided below. Each context snippet is marked with '--- Context from 'filename': ---'. Answer the user's question based *only* on this structured context. If the context doesn't contain enough information to answer, say so clearly.\n\n"
            f"Structured Context:\n{context}\n---\nUser Question: {current_input}\nAnswer:"
        )
        # st.write(f"DEBUG: Prompt length: {len(prompt)}") # Check prompt size

        # Call OpenAI's ChatCompletion API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant answering based *only* on the provided context.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=250,  # Increased slightly
            temperature=0.1,  # Slightly lower temp
        )
        answer = response.choices[0].message.content.strip()
        st.session_state.chat_history.append({"role": "assistant", "text": answer})

    except Exception as e:
        st.error("Error during chat processing: " + str(e))
        st.session_state.chat_history.append(
            {"role": "assistant", "text": f"An error occurred: {str(e)}"}
        )


# Text input widget using a form
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([4, 1])
    with col1:
        # Text input bound to session state, but NO on_change here
        st.text_input(
            "Type your question:", key="input_text", label_visibility="collapsed"
        )
    with col2:
        # The submit button for the form
        submitted = st.form_submit_button("Send")

    # Code here runs ONLY when the submit button is clicked
    if submitted:
        # Check if there's text in the state (captured by the form)
        if st.session_state.input_text and st.session_state.input_text.strip():
            send_message()  # Call send_message only on valid submission
        else:
            st.error("Please enter a question before sending.")
            # No need to call send_message if input is empty


# -------------------------------
# Display Chat History
# -------------------------------

st.markdown("#### Conversation")
for msg in reversed(st.session_state.chat_history):
    if msg["role"] == "user":
        st.markdown(
            f"<div style='text-align: right; color: lightblue;'><strong>You:</strong> {msg['text']}</div>",
            unsafe_allow_html=True,
        )
    else:  # Assistant or Error message
        color = (
            "lightcoral" if "error occurred" in msg["text"].lower() else "lightgreen"
        )
        st.markdown(
            f"<div style='text-align: left; color: {color};'><strong>Assistant:</strong> {msg['text']}</div>",
            unsafe_allow_html=True,
        )
