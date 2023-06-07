# pdf_chat_app.py
import os
import streamlit as st
from PyPDF2 import PdfReader
from apikey import apikey
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

class PDFChatApp:
    def __init__(self):
        # Setting up the OpenAI API
        os.environ['OPENAI_API_KEY'] = apikey

        # App environment
        st.title('🦜🔗Chat with PDF')
        

    def extract_text_from_pdf(self, pdf):
        pdf_reader = PdfReader(pdf)
        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text()

        return text

    def split_text_into_chunks(self, text):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text)
        return chunks

    def create_embeddings(self, chunks):
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        return knowledge_base

    def answer_question(self, knowledge_base, user_question):
        docs = knowledge_base.similarity_search(user_question)
        llm = OpenAI(temperature=0)
        chain = load_qa_chain(llm, chain_type="map_reduce", verbose=False)
        response = chain.run(input_documents=docs, question=user_question, verbose=False)
        return response

    def run(self):
        # Upload PDF
        pdf = st.file_uploader("Upload your PDF", type="pdf")

        # Extract the text
        if pdf:
            text = self.extract_text_from_pdf(pdf)
            chunks = self.split_text_into_chunks(text)
            knowledge_base = self.create_embeddings(chunks)

            user_question = st.text_input("Ask me questions from your PDF", key="user_query")

            if user_question:
                response = self.answer_question(knowledge_base, user_question)
                st.write(response)
