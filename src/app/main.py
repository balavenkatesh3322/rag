"""
Streamlit application for PDF-based Retrieval-Augmented Generation (RAG)
using Ollama + LangChain, with an integrated login page.

This application allows users to upload a PDF, process it,
and then ask questions about the content using a selected language model,
after successful authentication.
"""

import streamlit as st
import logging
import os
import tempfile
import shutil
import pdfplumber
import ollama
import warnings

# Suppress torch warning
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.classes.*")

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List, Tuple, Dict, Any, Optional

# Set protobuf environment variable to avoid error messages
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Define persistent directory for ChromaDB
PERSIST_DIRECTORY = os.path.join("data", "vectors")

# Streamlit page configuration
st.set_page_config(
    page_title="AIBOTS",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# --- Authentication Logic ---
def check_credentials(email, password):
    """
    Placeholder function to check credentials.
    In a real application, this would involve checking against a database
    or an authentication service.
    For this example, we'll use hardcoded credentials.
    """
    # IMPORTANT: Replace with secure credential checking in a real app!
    # For demonstration purposes only:
    valid_email = "user@example.com"
    valid_password = "password123"
    valid_access_key = "accesskey456"

    if (
        email == valid_email
        and password == valid_password
        # and access_key == valid_access_key
    ):
        return True
    return False


def login_page():
    """Displays the login page and handles authentication."""
    st.markdown(
        """
        <style>
            /* Tailwind-like utility classes for styling - simplified for Streamlit markdown */
            .login-container {
                background-color: #1f2937; /* slate-800 */
                padding: 2rem; /* p-8 */
                border-radius: 0.75rem; /* rounded-xl */
                box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04); /* shadow-2xl */
                width: 100%;
                max-width: 28rem; /* max-w-md */
                margin: auto; /* Center the container */
                border: 1px solid #374151; /* border-slate-700 */
            }
            .login-header {
                text-align: center;
                margin-bottom: 2rem; /* mb-8 */
            }
            .login-title {
                font-size: 1.875rem; /* text-3xl */
                font-weight: bold;
                color: #f3f4f6; /* slate-100 */
            }
            .login-subtitle {
                color: #9ca3af; /* slate-400 */
                margin-top: 0.5rem; /* mt-2 */
            }
            .input-label {
                display: block;
                font-size: 0.875rem; /* text-sm */
                font-weight: 500; /* font-medium */
                color: #d1d5db; /* slate-300 */
                margin-bottom: 0.25rem; /* mb-1 */
            }
            /* Streamlit's st.text_input and st.button will have their own styling,
               but we can try to influence the container and general text. */
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Use a container to apply custom styling if needed, though Streamlit's layout is primary
    with st.container():  # This acts somewhat like the outer div of the HTML
        # Centering the login form using columns
        col1_form, col2_form, col3_form = st.columns(
            [1, 2, 1]
        )  # Adjust ratios as needed

        with col2_form:
            st.markdown("<div class='login-header'>", unsafe_allow_html=True)
            # You can embed an SVG icon using st.markdown if needed, or use an emoji
            st.markdown(
                """
                <div style="display: inline-block; padding: 0.75rem; background-color: #4f46e5; border-radius: 9999px; margin-bottom: 1rem;">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" style="width: 2rem; height: 2rem; color: white;">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M15.75 5.25a3 3 0 0 1 3 3m3 0a6 6 0 0 1-7.029 5.912c-.563-.097-1.159.026-1.563.43L10.5 17.25H8.25v2.25H6v2.25H2.25v-2.818c0-.597.237-1.17.659-1.591l6.499-6.499c.404-.404.527-1 .43-1.563A6 6 0 1 1 21.75 8.25Z" />
                    </svg>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                "<h1 class='login-title'>Welcome to AIBOTS</h1>", unsafe_allow_html=True
            )
            st.markdown(
                "<p class='login-subtitle'>Sign in to access your dashboard.</p>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            with st.form("login_form", clear_on_submit=False):
                st.markdown(
                    "<label for='email' class='input-label'>Email Address</label>",
                    unsafe_allow_html=True,
                )
                email = st.text_input(
                    "Email Address",
                    placeholder="you@example.com",
                    key="email_input",
                    label_visibility="collapsed",
                )

                st.markdown(
                    "<label for='password' class='input-label' style='margin-top: 1rem;'>Password</label>",
                    unsafe_allow_html=True,
                )
                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="••••••••",
                    key="password_input",
                    label_visibility="collapsed",
                )

                # st.markdown(
                #     "<label for='accessKey' class='input-label' style='margin-top: 1rem;'>Access Key</label>",
                #     unsafe_allow_html=True,
                # )
                # access_key = st.text_input(
                #     "Access Key",
                #     placeholder="Enter your access key",
                #     key="access_key_input",
                #     label_visibility="collapsed",
                # )

                submitted = st.form_submit_button("Sign In", use_container_width=True)

                if submitted:
                    if check_credentials(email, password):
                        st.session_state["authenticated"] = True
                        st.session_state["user_email"] = (
                            email  # Optionally store user info
                        )
                        st.rerun()  # Rerun to reflect authenticated state
                    else:
                        st.error("Invalid email, password")

            st.markdown(
                "<div style='text-align: center; margin-top: 1.5rem;'>",
                unsafe_allow_html=True,
            )
            # st.markdown(
            #     "<a href='#' style='font-size: 0.875rem; color: #818cf8; text-decoration: none;'>Forgot Password?</a>",
            #     unsafe_allow_html=True,
            # )  # Style link
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div style="margin-top: 2.5rem; text-align: center;">
                <p style="font-size: 0.75rem; color: #6b7280;">&copy; 2025 AIBOTS. All rights reserved.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --- Main Application Logic (RAG App) ---
def extract_model_names(models_info: Any) -> Tuple[str, ...]:
    logger.info("Extracting model names from models_info")
    try:
        if hasattr(models_info, "models"):
            model_names = tuple(model.model for model in models_info.models)
        else:
            model_names = tuple()
        logger.info(f"Extracted model names: {model_names}")
        return model_names
    except Exception as e:
        logger.error(f"Error extracting model names: {e}")
        return tuple()


def create_vector_db(file_upload) -> Chroma:
    logger.info(f"Creating vector DB from file upload: {file_upload.name}")
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, file_upload.name)
    with open(path, "wb") as f:
        f.write(file_upload.getvalue())
    logger.info(f"File saved to temporary path: {path}")
    loader = UnstructuredPDFLoader(path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    logger.info("Document split into chunks")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    # Ensure PERSIST_DIRECTORY exists
    if not os.path.exists(PERSIST_DIRECTORY):
        os.makedirs(PERSIST_DIRECTORY)
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=f"pdf_{hash(file_upload.name)}",
    )
    logger.info("Vector DB created with persistent storage")
    shutil.rmtree(temp_dir)
    logger.info(f"Temporary directory {temp_dir} removed")
    return vector_db


def process_question(question: str, vector_db: Chroma, selected_model: str) -> str:
    logger.info(f"Processing question: {question} using model: {selected_model}")
    llm = ChatOllama(model=selected_model)
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate 2
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    response = chain.invoke(question)
    logger.info("Question processed and response generated")
    return response


@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    logger.info(f"Extracting all pages as images from file: {file_upload.name}")
    pdf_pages = []
    # Ensure file_upload is reset to the beginning if it's already been read
    file_upload.seek(0)
    with pdfplumber.open(file_upload) as pdf:
        pdf_pages = [page.to_image().original for page in pdf.pages]
    logger.info("PDF pages extracted as images")
    return pdf_pages


def delete_vector_db(vector_db: Optional[Chroma]) -> None:
    logger.info("Deleting vector DB")
    if vector_db is not None:
        try:
            vector_db.delete_collection()
            st.session_state.pop("pdf_pages", None)
            st.session_state.pop("file_upload", None)
            st.session_state.pop("vector_db", None)
            st.success("Collection and temporary files deleted successfully.")
            logger.info("Vector DB and related session state cleared")
            st.rerun()
        except Exception as e:
            st.error(f"Error deleting collection: {str(e)}")
            logger.error(f"Error deleting collection: {e}")
    else:
        st.error("No vector database found to delete.")
        logger.warning("Attempted to delete vector DB, but none was found")


def rag_application():
    """Main function to run the RAG Streamlit application after login."""
    st.subheader("🧠 AIBOTS", divider="gray", anchor=False)

    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state["authenticated"] = False
        # Clear other session state variables related to the RAG app
        keys_to_clear = [
            "messages",
            "vector_db",
            "use_sample",
            "pdf_pages",
            "file_upload",
            "user_email",
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    if "user_email" in st.session_state:
        st.sidebar.success(f"Logged in as: {st.session_state['user_email']}")

    models_info = ollama.list()
    available_models = extract_model_names(models_info)
    col1, col2 = st.columns([1.5, 2])

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None
    if "use_sample" not in st.session_state:
        st.session_state["use_sample"] = False

    selected_model = None  # Initialize selected_model
    if available_models:
        selected_model = col2.selectbox(
            "Pick a model available locally on your system ↓",
            available_models,
            key="model_select",
        )
    else:
        col2.warning(
            "No Ollama models found. Please ensure Ollama is running and models are installed."
        )

    use_sample = col1.toggle(
        "Use sample PDF (Scammer Agent Paper)", key="sample_checkbox"
    )

    if use_sample != st.session_state.get(
        "use_sample_previous_state", use_sample
    ):  # Check if toggle changed
        if st.session_state["vector_db"] is not None:
            logger.info("Toggle changed, deleting existing vector DB.")
            # vector_db.delete_collection() # This might error if collection name is not found
            st.session_state["vector_db"] = None  # Reset vector_db
            st.session_state["pdf_pages"] = None
        st.session_state["use_sample_previous_state"] = (
            use_sample  # Update previous state
        )

    if use_sample:
        sample_path = (
            "data/pdfs/sample/scammer-agent.pdf"  # Ensure this path is correct
        )
        # Create the directory if it doesn't exist and add a placeholder if the PDF is missing
        sample_dir = os.path.dirname(sample_path)
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
            logger.info(f"Created sample directory: {sample_dir}")

        if not os.path.exists(sample_path):
            # Create a dummy PDF for demonstration if the sample is missing
            try:
                from reportlab.pdfgen import canvas

                c = canvas.Canvas(sample_path)
                c.drawString(100, 750, "This is a dummy sample PDF for AIBOTS.")
                c.drawString(
                    100,
                    730,
                    "Please replace scammer-agent.pdf in data/pdfs/sample/ folder.",
                )
                c.save()
                logger.info(f"Created dummy sample PDF at: {sample_path}")
                col1.warning(
                    f"Sample PDF '{sample_path}' not found. A dummy PDF has been created. Please replace it."
                )
            except ImportError:
                logger.error("reportlab is not installed. Cannot create dummy PDF.")
                col1.error(
                    f"Sample PDF '{sample_path}' not found. Please create it or install reportlab to generate a dummy."
                )

        if os.path.exists(sample_path):
            if st.session_state["vector_db"] is None:
                with col1.status(
                    "Processing sample PDF...", expanded=True
                ) as status_sample:
                    try:
                        # Create a file-like object for UnstructuredPDFLoader and pdfplumber
                        with open(sample_path, "rb") as f_sample:
                            # For UnstructuredPDFLoader, pass the file path
                            loader = UnstructuredPDFLoader(file_path=sample_path)
                            data = loader.load()
                            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=7500, chunk_overlap=100
                            )
                            chunks = text_splitter.split_documents(data)

                            # Ensure PERSIST_DIRECTORY exists
                            if not os.path.exists(PERSIST_DIRECTORY):
                                os.makedirs(PERSIST_DIRECTORY)

                            st.session_state["vector_db"] = Chroma.from_documents(
                                documents=chunks,
                                embedding=OllamaEmbeddings(model="nomic-embed-text"),
                                persist_directory=PERSIST_DIRECTORY,
                                collection_name="sample_pdf_collection",  # Use a fixed name for sample
                            )
                        # For pdfplumber, reopen the file or pass the BytesIO object
                        with open(sample_path, "rb") as f_plumber:
                            st.session_state["pdf_pages"] = extract_all_pages_as_images(
                                f_plumber
                            )
                        status_sample.update(
                            label="Sample PDF processed!", state="complete"
                        )
                    except Exception as e:
                        status_sample.update(
                            label="Error processing sample PDF.", state="error"
                        )
                        col1.error(f"Error with sample PDF: {e}")
                        logger.error(f"Error processing sample PDF: {e}")
        # else: # This case is handled by the dummy PDF creation or error message above
        # col1.error("Sample PDF file not found.") # Already handled

    else:  # Not use_sample
        file_upload = col1.file_uploader(
            "Upload a PDF file ↓",
            type="pdf",
            accept_multiple_files=False,
            key="pdf_uploader",
        )
        if file_upload:
            # if (
            #     st.session_state.get("file_upload_id") != file_upload.id
            # ):  # New file uploaded
            #     if st.session_state["vector_db"] is not None:
            #         logger.info("New file uploaded, deleting existing vector DB.")
            #         # st.session_state["vector_db"].delete_collection() # This might error
            #         st.session_state["vector_db"] = None
            #     st.session_state["pdf_pages"] = None
            #     st.session_state["file_upload_id"] = file_upload.id  # Store new file id

            if st.session_state["vector_db"] is None:
                with col1.status(
                    "Processing uploaded PDF...", expanded=True
                ) as status_upload:
                    try:
                        st.session_state["vector_db"] = create_vector_db(file_upload)
                        st.session_state["pdf_pages"] = extract_all_pages_as_images(
                            file_upload
                        )
                        st.session_state["file_upload"] = (
                            file_upload  # Keep a reference if needed
                        )
                        status_upload.update(label="PDF processed!", state="complete")
                    except Exception as e:
                        status_upload.update(
                            label="Error processing PDF.", state="error"
                        )
                        col1.error(f"Error processing PDF: {e}")
                        logger.error(f"Error processing PDF: {e}")

    if "pdf_pages" in st.session_state and st.session_state["pdf_pages"]:
        zoom_level = col1.slider(
            "Zoom Level",
            min_value=100,
            max_value=1000,
            value=700,
            step=50,
            key="zoom_slider",
        )
        with col1:
            with st.container(height=410, border=True):
                for page_image in st.session_state["pdf_pages"]:
                    st.image(page_image, width=zoom_level)

    if col1.button(
        "⚠️ Delete collection and clear PDF",
        type="secondary",
        key="delete_button",
        use_container_width=True,
    ):
        delete_vector_db(st.session_state.get("vector_db"))
    st.markdown(
        """
                    <div style="margin-top: 2.5rem; text-align: center;">
                        <p style="font-size: 0.75rem; color: #6b7280;">&copy; 2025 AIBOTS. All rights reserved.</p>
                    </div>
                    """,
        unsafe_allow_html=True,
    )
    with col2:
        message_container = st.container(height=500, border=True)
        for i, message in enumerate(st.session_state["messages"]):
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        if prompt := st.chat_input("Enter a prompt here...", key="chat_input"):
            try:
                st.session_state["messages"].append({"role": "user", "content": prompt})
                with message_container.chat_message("user", avatar="😎"):
                    st.markdown(prompt)

                with message_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner(":green[processing...]"):
                        if (
                            st.session_state.get("vector_db") is not None
                            and selected_model
                        ):
                            response = process_question(
                                prompt, st.session_state["vector_db"], selected_model
                            )
                            st.markdown(response)
                            st.session_state["messages"].append(
                                {"role": "assistant", "content": response}
                            )
                        elif not selected_model:
                            st.warning("Please select a model to chat.")
                        else:  # No vector_db
                            st.warning(
                                "Please upload a PDF file or use the sample PDF to begin chat."
                            )

            except Exception as e:
                st.error(f"An error occurred: {e}", icon="⛔️")
                logger.error(f"Error processing prompt: {e}")
        else:
            if st.session_state.get("vector_db") is None and not use_sample:
                with message_container:
                    st.info("Upload a PDF file or use the sample PDF to begin chat...")


if __name__ == "__main__":
    # Initialize session state for authentication
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False  # Default to not authenticated

    if st.session_state["authenticated"]:
        rag_application()  # Show the main RAG app
    else:
        login_page()  # Show the login page
