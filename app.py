from turtle import Vec2D
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models.openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, user_template, bot_template
from dotenv import load_dotenv


def get_raw_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_obj = PdfReader(pdf)
        for page in pdf_obj.pages:
            text += page.extract_text()
    
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size = 600,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    return vectorstore

def get_conversation_chain(vector_store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_input(question):
    response = st.session_state.conversation({"question":question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Multiple PDF chat", page_icon=":books:")
    st.write(css,unsafe_allow_html=True)
    st.header("Chat with multiple pdfs :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    user_question = st.text_input("Please ask a question")

    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your docs and click 'submit'",accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Processing"):
                raw_text = get_raw_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vector_store)

if __name__ == "__main__":
    main()