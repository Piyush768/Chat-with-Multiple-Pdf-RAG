import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI


# Load environment variables
load_dotenv()

# Initialize the OpenAI model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9, max_tokens=500)

# Define the FAISS index path
faiss_index_path = "faiss_index"

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def process_pdf_text(raw_text):
    """Process and chunk the raw PDF text."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(raw_text)

def create_vectorstore(docs):
    """Create a FAISS vector store from document chunks."""
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(faiss_index_path)
    return vectorstore

def main():
    """Main Streamlit app logic."""
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon="📚")
    st.title("Chat with Multiple PDFs 📚")

    # Sidebar
    st.sidebar.subheader("Upload Your PDFs")
    pdf_docs = st.sidebar.file_uploader("Choose PDF files", accept_multiple_files=True)

    # Initialize placeholder for interaction
    main_placeholder = st.empty()

    # Button to process documents
    if st.sidebar.button("Process PDFs"):
        if not pdf_docs:
            st.sidebar.error("⚠️ Please upload at least one PDF file.")
            return

        with st.spinner("Processing PDFs..."):
            # Extract text from PDFs
            raw_text = get_pdf_text(pdf_docs)
            if not raw_text:
                st.error("⚠️ No text found in the PDFs.")
                return

            # Split the text into chunks
            st.sidebar.text("✂️ Splitting text into chunks...")
            docs = process_pdf_text(raw_text)
            if not docs:
                st.error("⚠️ No documents were created after splitting.")
                return

            # Create FAISS vector store
            vectorstore = create_vectorstore(docs)
            st.sidebar.success("PDFs processed and vector store created!")

    # Query input section
    query = main_placeholder.text_input("Ask a question based on the documents:")

    if query:
        # Load FAISS index if it exists
        if os.path.exists(faiss_index_path):
            try:
                embeddings = OpenAIEmbeddings()
                vectorstore = FAISS.load_local(faiss_index_path, embeddings)

                # Create a retrieval chain using the vector store
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                result = chain({"question": query}, return_only_outputs=True)

                # Display the answer
                st.header("✅ Answer")
                st.write(result.get("answer", "⚠️ No answer found."))

                # Display sources if available
                sources = result.get("sources", "").strip()
                if sources:
                    st.subheader("🔗 Sources:")
                    for source in sources.split("\n"):
                        st.write(f"🔹 {source}")

            except Exception as e:
                st.error(f"❌ Error retrieving answer: {e}")
        else:
            st.error("⚠️ FAISS index not found. Please process the PDFs first.")

if __name__ == "__main__":
    main()

