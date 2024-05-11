import streamlit as st
import os
import time
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from htmlTemplates import css, bot_template, user_template


if not os.path.exists('pdfFiles'):
   os.makedirs('pdfFiles')


if not os.path.exists('vectorDB'):
   os.makedirs('vectorDB')




if 'template' not in st.session_state:
   st.session_state.template = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative. Memorize all information in the .pdf files provided to answer questions.


   Context: {context}
   History: {history}


   User: {question}
   Chatbot:"""


if 'prompt' not in st.session_state:
   st.session_state.prompt = PromptTemplate(
       input_variables=["history", "context", "question"],
       template=st.session_state.template,
   )


if 'memory' not in st.session_state:
   st.session_state.memory = ConversationBufferMemory(
       memory_key="history",
       return_messages=True,
       input_key="question",
   )


if 'vectorstore' not in st.session_state:
   st.session_state.vectorstore = Chroma(persist_directory='vectorDb',
                                           embedding_function=OllamaEmbeddings(base_url='http://localhost:11434',
                                           model="llama3")
                                           )
  
if 'llm' not in st.session_state:
   st.session_state.llm = Ollama(base_url="http://localhost:11434",
                                 model="llama3",
                                 verbose=True,
                                 callback_manager=CallbackManager(
                                     [StreamingStdOutCallbackHandler()]),
                                 )
  
if 'chat_history' not in st.session_state:
   st.session_state.chat_history = []


st.title("Dialoguer - talk with your PDFs")


uploaded_file = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

st.write(css, unsafe_allow_html=True)



for i, message in enumerate(st.session_state.chat_history):
   with st.chat_message(message["role"]):
        if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}",message["message"]), unsafe_allow_html=True)
        else:
               st.write(bot_template.replace(
                    "{{MSG}}",message["message"]), unsafe_allow_html=True)

if uploaded_file is not None:
   st.text("File uploaded successfully")
   for x in uploaded_file:
    if not os.path.exists('pdfFiles/' + x.name):
        with st.status("Saving file..."):
            bytes_data = x.read()
            f = open('pdfFiles/' + x.name, 'wb')
            f.write(bytes_data)
            f.close()


            loader = PyPDFLoader('pdfFiles/' + x.name)
            data = loader.load()


            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200,
                length_function=len
            )


            all_splits = text_splitter.split_documents(data)


            st.session_state.vectorstore = Chroma.from_documents(
                documents = all_splits,
                embedding = OllamaEmbeddings(model = "llama3")
            )


            st.session_state.vectorstore.persist()


   st.session_state.retriever = st.session_state.vectorstore.as_retriever()


   if 'qa_chain' not in st.session_state:
       st.session_state.qa_chain = RetrievalQA.from_chain_type(
           llm=st.session_state.llm,
           chain_type='stuff',
           retriever=st.session_state.retriever,
           verbose=True,
           chain_type_kwargs={
               "verbose": True,
               "prompt": st.session_state.prompt,
               "memory": st.session_state.memory,
           }
       )


   if user_input := st.chat_input("You:", key="user_input"):
       user_message = {"role": "user_template", "message": user_input}
       st.session_state.chat_history.append(user_message)
       with st.chat_message("user_template"):
           st.write(user_template.replace(
                "{{MSG}}",user_input),unsafe_allow_html=True)


       with st.chat_message("bot_template"):
           with st.spinner("Assistant is typing..."):
               response = st.session_state.qa_chain(user_input)
           message_placeholder = st.empty()
           full_response = ""
           for chunk in response['result'].split():
               full_response += chunk + " "
               time.sleep(0.05)
               # Add a blinking cursor to simulate typing
               message_placeholder.write(bot_template.replace(
                   "{{MSG}}",full_response + "▌"), unsafe_allow_html=True)
           message_placeholder.write(bot_template.replace(
               "{{MSG}}",full_response + "▌"), unsafe_allow_html=True)


       chatbot_message = {"role": "Agent", "message": response['result']}
       st.session_state.chat_history.append(chatbot_message)


else:
   st.write("Please upload a PDF file to start the chatbot")




