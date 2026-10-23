import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from htmlTemplate import css, bot_template, user_template 

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size=1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_pdf_text(pdf_docs):
    full_text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            full_text += page.extract_text() or ""

    return full_text

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorestore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vectorestore

def get_conversation_chain(vectorestore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain =  ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorestore.as_retriever(),
        memory = memory
    )
    return conversation_chain

def handle_user_question(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.write(response)

def main():
    load_dotenv()

    st.write(css, unsafe_allow_html=True)
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("Chat with Multiple PDFs :books:")
    user_question = st.text_input("Start chatting with them!")
    if user_question:
        handle_user_question(user_question)

    st.write(user_template.replace("{{MSG}}", "Hello Robot"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello Human"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on Precess",
                         type=["pdf"], accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Precessing ..."):
                # get the pdf text
                raw_text = get_pdf_text(pdf_docs)
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # create the vector db
                get_vectorstore(text_chunks)
                st.write(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorestore)


if __name__ == "__main__":
    main()