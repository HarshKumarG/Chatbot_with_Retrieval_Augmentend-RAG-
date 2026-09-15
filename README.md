
# 🤖 Chatbot with Retrieval Augmented Generation (RAG)

A document-based AI chatbot that allows users to upload PDF files and ask questions about their content using Retrieval Augmented Generation (RAG).

This project combines document processing, text embeddings, vector databases, and Large Language Models (LLMs) to retrieve relevant information from uploaded documents and generate context-aware answers.

---

## 📌 Project Overview

Large Language Models are powerful tools for answering questions, but they may not have access to information contained in a user's private documents.

This project solves that problem by building a **PDF Question-Answering Chatbot using Retrieval Augmented Generation (RAG)**.

Users can upload PDF documents and ask questions in natural language. The application processes the documents, converts their content into numerical embeddings, stores those embeddings in a FAISS vector database, and retrieves relevant document sections when a question is asked.

The retrieved information is then provided as context to the language model, which generates a relevant and informative response.

The application is developed using Python, Streamlit, LangChain, OpenAI Embeddings, and FAISS.

---

## 🚀 Project Highlights

- 📄 Upload and process PDF documents.
- 💬 Ask questions about uploaded documents.
- 🔍 Retrieve relevant information using semantic search.
- 🧠 Generate context-aware answers using an LLM.
- 📚 Use OpenAI embeddings for document representation.
- ⚡ Store and search document embeddings using FAISS.
- ✂️ Split large documents into smaller text chunks.
- 🖥️ Interactive web interface using Streamlit.
- 🤖 Build a practical Generative AI application.
- 🔐 Process user-provided documents within the application workflow.

---

## 🎯 Problem Statement

Traditional document search often depends on exact keyword matching.

For example, if a document contains:

> "The company provides employees with 20 days of annual leave."

A user may ask:

> "How many vacation days do employees receive?"

A keyword-based search may not find the correct information because the words "vacation days" and "annual leave" are different.

A RAG-based chatbot uses semantic similarity to retrieve relevant information, allowing the system to answer questions using the meaning of the content rather than relying only on exact keyword matches.

---

## 🧠 What is Retrieval Augmented Generation (RAG)?

Retrieval Augmented Generation is a technique that combines:

1. **Retrieval** – Search for relevant information from a knowledge base.
2. **Augmentation** – Add the retrieved information to the LLM prompt as context.
3. **Generation** – Generate an answer using the retrieved context.

Instead of relying only on the model's pre-trained knowledge, the chatbot retrieves relevant information from the uploaded documents.

### RAG Architecture

```text
                 ┌──────────────────────┐
                 │      User Uploads    │
                 │      PDF Document    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │    PDF Document      │
                 │       Loading        │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │    Text Extraction   │
                 │      from PDF        │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │    Text Splitting    │
                 │  Recursive Splitter  │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   OpenAI Embeddings  │
                 │   Text → Vectors     │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │     FAISS Vector     │
                 │       Database       │
                 └──────────┬───────────┘
                            │
                            │
                    User Asks Question
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Question Embedding │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │  Similarity Search   │
                 │  Retrieve Relevant   │
                 │   Document Chunks    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │    Prompt + Context  │
                 │      + Question      │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │    OpenAI LLM        │
                 │   Answer Generation  │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Final Answer to    │
                 │        User          │
                 └──────────────────────┘
```

---

## ✨ Key Features

### 1. PDF Document Upload

Users can upload PDF documents through the Streamlit interface.

The application extracts text from the uploaded files and prepares it for further processing.

### 2. Document Text Extraction

PDF content is extracted using PyPDF2.

```python
from PyPDF2 import PdfReader

reader = PdfReader("document.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text()
```

The extracted text is used as the input for the document processing pipeline.

### 3. Text Chunking

Large documents are divided into smaller chunks using a text splitter.

This improves retrieval by allowing the system to search for relevant sections instead of processing the entire document at once.

Example:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_text(text)
```

**Why chunking is important:**

- Large documents may exceed the model's context window.
- Smaller chunks improve retrieval granularity.
- Relevant sections can be passed to the LLM.
- Chunk overlap helps preserve context between sections.

### 4. Text Embeddings

The application converts text chunks into numerical vectors using OpenAI Embeddings.

Embeddings represent the semantic meaning of text in numerical form.

For example:

```text
"How do I reset my password?"

and

"What are the steps to change my password?"
```

These sentences have similar meanings and can have similar embedding representations.

Example:

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)
```

> Use the embedding model configured in your actual project.

### 5. FAISS Vector Database

FAISS is used to store and search the document embeddings.

FAISS stands for **Facebook AI Similarity Search**.

It enables efficient similarity search across numerical vectors.

The document chunks are embedded and stored in a FAISS index.

When the user asks a question:

1. The question is converted into an embedding.
2. FAISS searches for similar document vectors.
3. The most relevant chunks are retrieved.
4. The retrieved chunks are passed to the LLM.

Example:

```python
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_texts(
    chunks,
    embeddings
)
```

### 6. Retrieval Augmented Generation

The retrieved document chunks are used as context for the language model.

The LLM generates an answer based on the retrieved content.

Conceptually:

```text
User Question
      +
Retrieved Document Context
      ↓
Prompt
      ↓
Large Language Model
      ↓
Generated Answer
```

This approach helps the chatbot answer questions about documents that were not part of the model's original training data.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| Streamlit | Interactive web application |
| PyPDF2 | Extract text from PDF documents |
| LangChain | Build the RAG application pipeline |
| OpenAI Embeddings | Convert text into numerical vectors |
| FAISS | Vector storage and similarity search |
| OpenAI LLM | Generate answers from retrieved context |
| RecursiveCharacterTextSplitter | Split documents into chunks |
| Google Colab | Development and experimentation |
| Git & GitHub | Version control and project hosting |

---

## 🔄 Complete RAG Pipeline

### Step 1: Document Ingestion

The user uploads a PDF document.

The application reads the document and extracts its text.

### Step 2: Text Preprocessing

The extracted text is cleaned and prepared for chunking.

### Step 3: Document Chunking

The text is divided into smaller overlapping chunks.

### Step 4: Embedding Generation

Each chunk is converted into an embedding vector.

### Step 5: Vector Storage

The embeddings are stored in a FAISS vector database.

### Step 6: User Query

The user asks a question in natural language.

### Step 7: Query Embedding

The question is converted into an embedding using the same embedding model.

### Step 8: Similarity Search

FAISS retrieves the most relevant document chunks.

### Step 9: Prompt Construction

The retrieved chunks are combined with the user's question.

### Step 10: Answer Generation

The LLM generates a response using the retrieved context.

---

## 💻 Example Implementation

The following code demonstrates the core RAG workflow.

### Import Libraries

```python
import os

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
```

### Create Text Chunks

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_text(text)
```

### Generate Embeddings

```python
embeddings = OpenAIEmbeddings()
```

### Create FAISS Vector Store

```python
vectorstore = FAISS.from_texts(
    chunks,
    embedding=embeddings
)
```

### Retrieve Relevant Documents

```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)
```

### Search for Relevant Context

```python
docs = retriever.invoke(
    "What is the main topic of this document?"
)

for doc in docs:
    print(doc.page_content)
```

### Generate an Answer

The retrieved content is supplied to the language model along with the user's question.

The exact implementation depends on the LangChain version and LLM configuration used in the project.

---

## 🖥️ Streamlit Application

The project includes a Streamlit interface that allows users to interact with the chatbot.

### User Workflow

1. Open the application.
2. Upload a PDF document.
3. Click the processing or submit button.
4. Ask questions about the uploaded document.
5. View the generated AI response.

### Example Interaction

```text
User:
What is the main objective of this document?

Chatbot:
The main objective of the document is to explain
the key concepts, methods, and information
discussed in the uploaded PDF.
```

The actual answer depends on the uploaded document and retrieved context.

---

## 📂 Project Structure

```text
Chatbot_with_Retrieval_Augmentend-RAG-/
│
├── Chatbot_with_Retrieval_Augmentend(RAG).ipynb
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
└── assets/
    └── screenshots/
```

> Update the structure according to the actual files available in your GitHub repository.

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/HarshKumarG/Chatbot_with_Retrieval_Augmentend-RAG-.git
```

### 2. Navigate to the Project Directory

```bash
cd Chatbot_with_Retrieval_Augmentend-RAG-
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Mac/Linux:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure OpenAI API Key

The application requires an OpenAI API key if your implementation uses OpenAI embeddings and an OpenAI language model.

Create a `.env` file:

```text
OPENAI_API_KEY=your_openai_api_key
```

Or configure the key through your environment variables.

> Never commit your actual API key to GitHub.

### 6. Run the Application

If your project contains `app.py`:

```bash
streamlit run app.py
```

If your project is implemented in Google Colab, open the notebook and execute the cells in order.

---

## 📦 Requirements

Example `requirements.txt`:

```text
streamlit
pypdf2
langchain
langchain-community
langchain-openai
faiss-cpu
openai
python-dotenv
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

> Dependency names and versions should match your actual project implementation.

---

## 🔐 Environment Variables

The project uses an OpenAI API key for embeddings and/or LLM access.

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | API key used to access OpenAI services |

Example:

```python
import os

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
```

Keep your API credentials private and do not upload them to GitHub.

---

## 📊 Technical Architecture

### Document Processing Layer

Responsible for:

- PDF upload.
- Text extraction.
- Text cleaning.
- Chunk creation.

### Embedding Layer

Responsible for:

- Converting text chunks into vectors.
- Converting user queries into vectors.
- Maintaining consistent embedding representation.

### Vector Database Layer

Responsible for:

- Storing document embeddings.
- Performing similarity search.
- Retrieving relevant chunks.

### LLM Generation Layer

Responsible for:

- Receiving the user's question.
- Receiving retrieved document context.
- Generating a natural-language response.

### User Interface Layer

Responsible for:

- Uploading documents.
- Accepting user questions.
- Displaying chatbot responses.

---

## 🎯 Business Use Cases

This RAG chatbot can be adapted for several real-world applications.

### 1. Document Question Answering

Users can ask questions about lengthy PDF documents without manually reading every page.

### 2. Research Assistant

Researchers can upload papers, reports, or study material and retrieve relevant information.

### 3. Educational Assistant

Students can upload study notes, textbooks, and academic documents to ask questions.

### 4. Business Knowledge Assistant

Organizations can use RAG systems to search internal documentation, policies, and reports.

### 5. Customer Support

Support teams can retrieve information from product manuals and knowledge bases.

### 6. Legal and Policy Document Search

Users can search through policies, agreements, and other documents.

> For legal, medical, or other high-stakes uses, answers should be verified against the original documents and reviewed by qualified professionals.

---

## ⚠️ Limitations

The current system has several limitations:

- Answer quality depends on the quality of the uploaded PDF.
- Scanned PDFs may require OCR for text extraction.
- Incorrect or incomplete text extraction can affect retrieval.
- The system may retrieve irrelevant chunks.
- The LLM can still generate incorrect answers.
- Large documents may require more processing time and memory.
- API usage may incur costs.
- The application may not maintain conversation history unless implemented.
- The system may not support multiple documents unless configured for it.

---

## 🔮 Future Improvements

### 1. Conversational Memory

Add chat history so users can ask follow-up questions.

### 2. Multi-PDF Support

Allow users to upload multiple documents and query them together.

### 3. Improved Retrieval

Use hybrid search combining keyword-based and semantic search.

### 4. Reranking

Add a reranking model to improve the relevance of retrieved chunks.

### 5. Source Citations

Display the PDF page number and source chunk used to generate each answer.

### 6. Advanced Embeddings

Experiment with different embedding models to improve semantic retrieval.

### 7. Document OCR

Add OCR support for scanned documents and image-based PDFs.

### 8. Cloud Deployment

Deploy the application using Streamlit Community Cloud or another cloud platform.

### 9. Authentication

Add secure login and access control for private documents.

### 10. Evaluation

Measure retrieval quality and answer quality using metrics such as:

- Precision@K
- Recall@K
- Retrieval hit rate
- Answer faithfulness
- Context relevance

---

## 📚 Learning Outcomes

Through this project, I gained practical experience in:

- Python programming.
- Generative AI application development.
- Retrieval Augmented Generation (RAG).
- Large Language Models (LLMs).
- LangChain.
- PDF document processing.
- Text chunking and preprocessing.
- OpenAI embeddings.
- Vector databases.
- FAISS similarity search.
- Prompt engineering.
- Streamlit application development.
- API integration.
- Git and GitHub.
- Building document-based AI assistants.

---

## 🚀 Future Scope

This project can be extended into a production-ready AI knowledge assistant by adding:

- Advanced retrieval strategies.
- Hybrid search.
- Reranking.
- Conversational memory.
- Multiple document support.
- Source attribution.
- Evaluation pipelines.
- Cloud deployment.
- Authentication and access control.
- Monitoring and logging.

---

## 👨‍💻 Author

**Harsh Kumar Gupta**

M.Sc. Computer Science – Specialization in Data Science

University of Mumbai

### Connect With Me

- 💼 LinkedIn: [Harsh Kumar Gupta](https://linkedin.com/in/harsh-kumar-gupta-8490a21ba)
- 🐙 GitHub: [HarshKumarG](https://github.com/HarshKumarG)

---

## ⭐ Support

If you found this project useful, consider giving the repository a ⭐ star.

Thank you for visiting this project!
