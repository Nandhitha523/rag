import os
import tempfile
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from transformers import pipeline

# Set up Hugging Face model for summarization/rephrasing
def get_huggingface_pipeline():
    model_name = "facebook/bart-large-cnn"
    pipe = pipeline(
        task="text2text-generation",
        model=model_name,
        tokenizer=model_name,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )
    return pipe

# Load PDF Documents (Fixed File Handling)
def load_documents(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())  # Save file to disk
            temp_file_path = temp_file.name  # Get the file path

        loader = PyPDFLoader(temp_file_path)
        documents.extend(loader.load())

        os.remove(temp_file_path)  # Cleanup after processing ✅ Fixes `PermissionError`
    return documents

# Streamlit UI
st.title("🔍 RAG Chatbot")

uploaded_files = st.file_uploader("📂 Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.write("📄 Processing documents...")
    documents = load_documents(uploaded_files)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings()
    faiss_index = FAISS.from_documents(documents, embeddings)

    st.success("✅ Documents indexed successfully!")

    # Query input
    user_query = st.text_input("💬 Ask a question:")

    if st.button("🔎 Search") and user_query:
        results = faiss_index.similarity_search(user_query, k=1)  # Get only 1 best match

        if results:
            best_result = results[0].page_content  # Extract the most relevant text
            
            # **Generate a natural answer using BART model**
            hf_pipeline = get_huggingface_pipeline()
            refined_answer = hf_pipeline(best_result)[0]['generated_text']

            st.write("### 📌 Answer:")
            st.write(refined_answer)  # Display the refined response
        else:
            st.warning("⚠️ No relevant results found. Try a different question!")
