# Chat with Multiple PDFs RAG

This project allows users to upload multiple PDF documents, extract their content, and then interact with the content by asking questions. The app uses advanced AI models to retrieve the most relevant information from the documents based on user queries.

### Key Features:
- Upload multiple PDF files.
- Automatically extract and process the content from PDFs.
- Split the text into chunks to handle larger documents.
- Store the processed content in a FAISS index for fast retrieval.
- Query the documents and get answers based on the content.
- Display relevant sources for the retrieved answer.

### Tech Stack:
- **Streamlit**: For building the interactive web app.
- **PyPDF2**: For extracting text from PDF documents.
- **Langchain**: For processing and embedding document content.
- **OpenAI API**: For generating embeddings and answering questions.
- **FAISS**: For efficient similarity search on the processed documents.
  
## Setup

### Prerequisites:
- Python 3.7 or higher
- OpenAI API key (to generate embeddings and responses)
- Install the required dependencies via `requirements.txt`

### Steps to Set Up:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/chat-with-multiple-pdfs.git
   cd chat-with-multiple-pdfs
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For macOS/Linux
   venv\Scripts\activate     # For Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set your OpenAI API key**:
   Create a `.env` file in the root of the project and add your OpenAI API key:
   ```env
   OPENAI_API_KEY=your-openai-api-key
   ```

5. **Run the app**:
   ```bash
   streamlit run app.py
   ```

6. **Upload PDFs**:
   Once the app is running, open it in your browser, and upload your PDF files via the sidebar.

### Usage:
- After uploading PDFs, click "Process PDFs" to extract and chunk the document text.
- Enter a query in the input field to ask questions related to the uploaded documents.
- The app will provide an answer along with relevant sources from the documents.

### Dependencies:
```txt
streamlit
langchain
langchain_openai
PyPDF2
openai
faiss-cpu
python-dotenv
```
