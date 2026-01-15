import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
import os
from datetime import datetime

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from typing import Optional, List
from transformers import pipeline
import torch


class HuggingFaceLLM(LLM):
    """Custom LLM wrapper for HuggingFace models"""

    pipeline: any = None

    def __init__(self, model_name="google/flan-t5-base"):
        super().__init__()
        self.pipeline = pipeline(
            "text2text-generation",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1,
            max_length=512
        )

    @property
    def _llm_type(self) -> str:
        return "huggingface"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.pipeline(prompt, max_length=512, do_sample=False)[0]['generated_text']
        return response


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    chunk_size = 10000
    chunk_overlap = 1000
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    """Create vector store using local embeddings"""
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


@st.cache_resource
def load_model(model_name):
    """Load and cache the model"""
    if model_name == "FLAN-T5 Small (fastest)":
        return HuggingFaceLLM(model_name="google/flan-t5-small")
    elif model_name == "FLAN-T5 Base (balanced)":
        return HuggingFaceLLM(model_name="google/flan-t5-base")
    elif model_name == "FLAN-T5 Large (best quality)":
        return HuggingFaceLLM(model_name="google/flan-t5-large")
    else:
        return HuggingFaceLLM(model_name="google/flan-t5-base")


def get_conversational_chain(model_name):
    """Create QA chain with local model"""
    prompt_template = """
    Answer the question based on the provided context. Be detailed and accurate.

    If the answer is not in the context, say "I cannot find this information in the provided documents."

    Context: {context}

    Question: {question}

    Answer:
    """

    model = load_model(model_name)

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question, model_name, pdf_docs, conversation_history):
    """Process user question and generate response"""
    if pdf_docs is None:
        st.warning("Please upload PDF files before asking questions.")
        return

    try:
        # Get embeddings
        embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        # Load vector store
        new_db = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )

        # Search for relevant documents
        docs = new_db.similarity_search(user_question, k=3)

        # Get conversational chain
        with st.spinner("Generating answer..."):
            chain = get_conversational_chain(model_name)

            # Generate response
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )

        user_question_output = user_question
        response_output = response['output_text']
        pdf_names = [pdf.name for pdf in pdf_docs] if pdf_docs else []

        # Add to conversation history
        conversation_history.append((
            user_question_output,
            response_output,
            model_name,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            ", ".join(pdf_names)
        ))

        # Display current Q&A
        st.markdown(
            f"""
            <style>
                .chat-message {{
                    padding: 1.5rem;
                    border-radius: 0.5rem;
                    margin-bottom: 1rem;
                    display: flex;
                }}
                .chat-message.user {{
                    background-color: #2b313e;
                }}
                .chat-message.bot {{
                    background-color: #475063;
                }}
                .chat-message .avatar {{
                    width: 20%;
                }}
                .chat-message .avatar img {{
                    max-width: 78px;
                    max-height: 78px;
                    border-radius: 50%;
                    object-fit: cover;
                }}
                .chat-message .message {{
                    width: 80%;
                    padding: 0 1.5rem;
                    color: #fff;
                }}
            </style>
            <div class="chat-message user">
                <div class="avatar">
                    <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
                </div>    
                <div class="message">{user_question_output}</div>
            </div>
            <div class="chat-message bot">
                <div class="avatar">
                    <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp">
                </div>
                <div class="message">{response_output}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Display conversation history (excluding the last item which is already shown)
        if len(conversation_history) > 1:
            for question, answer, model, timestamp, pdf_name in reversed(conversation_history[:-1]):
                st.markdown(
                    f"""
                    <div class="chat-message user">
                        <div class="avatar">
                            <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
                        </div>    
                        <div class="message">{question}</div>
                    </div>
                    <div class="chat-message bot">
                        <div class="avatar">
                            <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp">
                        </div>
                        <div class="message">{answer}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Provide CSV download option
        if len(st.session_state.conversation_history) > 0:
            df = pd.DataFrame(
                st.session_state.conversation_history,
                columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"]
            )
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv"><button>Download conversation history as CSV file</button></a>'
            st.sidebar.markdown(href, unsafe_allow_html=True)
            st.markdown("To download the conversation, click the Download button in the sidebar.")

        st.snow()

    except Exception as e:
        st.error(f"Error processing question: {str(e)}")


def main():
    st.set_page_config(page_title="Chat with multiple PDFs (Local)", page_icon=":books:")
    st.header("Chat with multiple PDFs - Free Local Version :books:")

    # Initialize session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Sidebar
    with st.sidebar:
        st.title("Settings")

        # Model selection
        model_name = st.radio(
            "Select Local Model:",
            ("FLAN-T5 Small (fastest)",
             "FLAN-T5 Base (balanced)",
             "FLAN-T5 Large (best quality)")
        )


        st.divider()

        # Control buttons
        col1, col2 = st.columns(2)
        clear_button = col1.button("Clear Last")
        reset_button = col2.button("Reset All")

        if reset_button:
            st.session_state.conversation_history = []
            if os.path.exists("faiss_index"):
                import shutil
                shutil.rmtree("faiss_index")
            st.success("Reset complete!")
            st.rerun()

        if clear_button:
            if len(st.session_state.conversation_history) > 0:
                st.session_state.conversation_history.pop()
                st.success("Last conversation cleared!")
            else:
                st.warning("No conversation to clear.")

        st.divider()

        # PDF upload
        pdf_docs = st.file_uploader(
            "Upload your PDF Files",
            accept_multiple_files=True,
            type=['pdf']
        )

        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    try:
                        # Extract text from PDFs
                        raw_text = get_pdf_text(pdf_docs)

                        if not raw_text.strip():
                            st.warning("PDFs appear to be empty or contain no extractable text")
                        else:
                            # Create text chunks
                            text_chunks = get_text_chunks(raw_text)

                            # Create vector store
                            get_vector_store(text_chunks)

                            st.success(f"Successfully processed {len(pdf_docs)} PDF(s)!")
                            st.info(f"Created {len(text_chunks)} text chunks for searching.")
                    except Exception as e:
                        st.error(f"Error processing PDFs: {str(e)}")
            else:
                st.warning("Please upload PDF files before processing.")

    # Main area - Question input
    user_question = st.text_input("Ask a question from the PDF files:")

    if user_question:
        if not os.path.exists("faiss_index"):
            st.warning("Please upload and process PDF files first!")
        else:
            user_input(
                user_question,
                model_name,
                pdf_docs,
                st.session_state.conversation_history
            )


if __name__ == "__main__":
    main()