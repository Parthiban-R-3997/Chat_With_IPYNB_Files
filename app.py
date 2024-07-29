import streamlit as st
import os
import tempfile
import time
import nbformat
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

st.set_page_config(page_title="Chat with Notebooks", page_icon=":books:")

st.title("Chat Gemini Document Q&A with Jupyter Notebooks")

# Custom prompt template
custom_context_input = """
<context>
{context}
</context>
Questions:{input}
"""

# Default prompt template
default_prompt_template = """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
</context>
Questions:{input}
"""

def load_notebook(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)
    return notebook

def extract_text_from_notebook(notebook):
    text = []
    for cell in notebook.cells:
        if cell.cell_type == 'markdown':
            text.append(cell.source)
        elif cell.cell_type == 'code':
            text.append(cell.source)
            if 'outputs' in cell:
                for output in cell.outputs:
                    if output.output_type == 'stream':
                        text.append(output.text)
                    elif output.output_type == 'execute_result' and 'data' in output:
                        text.append(output.data.get('text/plain', ''))
    return "\n".join(text)

def vector_embedding(ipynb_files):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    documents = []
    for ipynb_file in ipynb_files:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ipynb") as tmp_file:
            tmp_file.write(ipynb_file.getvalue())
            tmp_file_path = tmp_file.name

        # Load the .ipynb file from the temporary file path
        notebook = load_notebook(tmp_file_path)
        text = extract_text_from_notebook(notebook)
        # Create a Document object instead of using plain text
        documents.append(Document(page_content=text))

        # Remove the temporary file
        os.remove(tmp_file_path)

    # Ensure documents are properly segmented or chunked
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    try:
        segmented_documents = st.session_state.text_splitter.split_documents(documents)
        st.session_state.final_documents = segmented_documents

        if st.session_state.final_documents:
            # Embedding using FAISS
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
            st.success("Document embedding is completed!")
        else:
            st.warning("No documents found to embed.")
    
    except Exception as e:
        st.error(f"Error splitting or embedding documents: {str(e)}")
        st.session_state.final_documents = []  # Handle empty documents or retry

# Define model options for Gemini
model_options = [
  "gemini-1.5-flash",
  "gemini-1.5-pro",
  "gemini-1.0-pro"
]

# Sidebar elements
with st.sidebar:
    st.header("Configuration")
    st.markdown("Enter your API key below:")
    google_api_key = st.text_input("Enter your Google API Key", type="password", help="Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)")
    selected_model = st.selectbox("Select Gemini Model", model_options)
    os.environ["GOOGLE_API_KEY"] = str(google_api_key)
    
    st.markdown("Upload your .ipynb files:")
    uploaded_files = st.file_uploader("Choose .ipynb files", accept_multiple_files=True, type="ipynb")

    # Custom prompt text areas
    custom_prompt_template = st.text_area("Custom Prompt Template", placeholder="Enter your custom prompt here...(optional)")

    if st.button("Start Document Embedding"):
        if uploaded_files:
            vector_embedding(uploaded_files)
            st.success("Vector Store DB is Ready")
        else:
            st.warning("Please upload at least one .ipynb file.")

# Main section for question input and results
prompt1 = st.text_area("Enter Your Question From Documents")

if prompt1 and "vectors" in st.session_state:
    if custom_prompt_template:
        custom_prompt = custom_prompt_template + custom_context_input
        prompt = ChatPromptTemplate.from_template(custom_prompt)
    else:
        prompt = ChatPromptTemplate.from_template(default_prompt_template)
    
    llm = ChatGoogleGenerativeAI(model=selected_model, temperature=0.3)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    st.write("Response time:", time.process_time() - start)
    st.write(response['answer'])

    # With a Streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")