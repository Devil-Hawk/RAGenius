import React, { useState, useCallback } from "react";
import axios from "axios";
import { useDropzone } from "react-dropzone";
import { motion } from "framer-motion";
import { FaUpload, FaPaperPlane } from "react-icons/fa";

function App() {
  const [files, setFiles] = useState([]);
  const [uploadStatus, setUploadStatus] = useState("");
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [errorMsg, setErrorMsg] = useState("");

  // Drag & drop file handler
  const onDrop = useCallback((acceptedFiles) => {
    setFiles(acceptedFiles);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

  // Upload files to the backend
  const uploadFiles = async () => {
    if (!files || files.length === 0) {
      setErrorMsg("No files selected for upload.");
      return;
    }
    setErrorMsg("");
    const formData = new FormData();
    files.forEach((file) => {
      formData.append("files", file);
    });
    try {
      const res = await axios.post("http://localhost:8000/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setUploadStatus(`Uploaded ${res.data.files.length} file(s) successfully.`);
    } catch (err) {
      setErrorMsg("File upload failed: " + (err.response?.data?.error || "Unknown error"));
    }
  };

  // Send a chat message to the backend and get a response
  const sendMessage = async () => {
    if (!input.trim()) return;
    setErrorMsg("");
    const newMessages = [...messages, { role: "user", text: input }];
    setMessages(newMessages);
    const currentInput = input;
    setInput("");
    try {
      const formData = new FormData();
      formData.append("question", currentInput);
      const res = await axios.post("http://localhost:8000/ask", formData);
      const answer = res.data.answer;
      setMessages([...newMessages, { role: "assistant", text: answer }]);
    } catch (err) {
      const errorText = "Error: " + (err.response?.data?.error || "Unknown error");
      setMessages([...newMessages, { role: "assistant", text: errorText }]);
      setErrorMsg(errorText);
    }
  };

  // Send on Enter key press
  const handleKeyPress = (e) => {
    if (e.key === "Enter") {
      sendMessage();
    }
  };

  // Dark theme styling
  const darkStyle = {
    backgroundColor: "#121212",
    color: "#ffffff",
    minHeight: "100vh",
    padding: "20px",
  };

  const chatBoxStyle = {
    border: "1px solid #444",
    borderRadius: "8px",
    width: "80%",
    maxWidth: "600px",
    margin: "0 auto",
    padding: "10px",
    height: "60vh",
    overflowY: "auto",
    backgroundColor: "#1e1e1e",
  };

  const messageStyle = (role) => ({
    textAlign: role === "user" ? "right" : "left",
    margin: "10px 0",
  });

  const messageBubbleStyle = (role) => ({
    display: "inline-block",
    padding: "8px 12px",
    borderRadius: "8px",
    backgroundColor: role === "user" ? "#333" : "#444",
    maxWidth: "70%",
    wordWrap: "break-word",
  });

  return (
    <div style={darkStyle}>
      <motion.h1
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        style={{ textAlign: "center", marginBottom: "20px" }}
      >
        RAGenius
      </motion.h1>

      {/* File Upload Section */}
      <div style={{ textAlign: "center", marginBottom: "20px" }}>
        <div
          {...getRootProps()}
          style={{
            border: "2px dashed #555",
            padding: "20px",
            borderRadius: "8px",
            backgroundColor: "#1e1e1e",
            cursor: "pointer",
            color: "#fff",
          }}
        >
          <input {...getInputProps()} />
          {isDragActive ? (
            <p>Drop the files here ...</p>
          ) : (
            <p>Drag & drop files here, or click to select files</p>
          )}
        </div>
        <button
          onClick={uploadFiles}
          style={{
            marginTop: "10px",
            padding: "10px 20px",
            border: "none",
            borderRadius: "8px",
            backgroundColor: "#007bff",
            color: "#fff",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            gap: "8px",
            margin: "10px auto",
          }}
        >
          <FaUpload /> Upload Files
        </button>
        {uploadStatus && <p style={{ color: "lightgreen" }}>{uploadStatus}</p>}
        {errorMsg && <p style={{ color: "red" }}>{errorMsg}</p>}
      </div>

      {/* Chat Section */}
      <div style={chatBoxStyle}>
        {messages.map((msg, index) => (
          <div key={index} style={messageStyle(msg.role)}>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              style={messageBubbleStyle(msg.role)}
            >
              {msg.text}
            </motion.div>
          </div>
        ))}
      </div>

      {/* Input Section */}
      <div style={{ textAlign: "center", marginTop: "20px" }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your question..."
          style={{
            width: "70%",
            maxWidth: "500px",
            padding: "10px",
            borderRadius: "8px",
            border: "1px solid #555",
            backgroundColor: "#1e1e1e",
            color: "#fff",
          }}
        />
        <button
          onClick={sendMessage}
          style={{
            marginLeft: "10px",
            padding: "10px 20px",
            border: "none",
            borderRadius: "8px",
            backgroundColor: "#28a745",
            color: "#fff",
            cursor: "pointer",
            display: "inline-flex",
            alignItems: "center",
            gap: "8px",
          }}
        >
          <FaPaperPlane /> Send
        </button>
      </div>
    </div>
  );
}

export default App;
