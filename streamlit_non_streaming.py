import os
import sys

import streamlit as st
from langchain import PromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.chains import RetrievalQA
from langchain.document_loaders import OnlinePDFLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.llms import Ollama
from langchain.vectorstores import Chroma



class SuppressStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


if 'qa_chain' not in st.session_state:
    st.session_state['qa_chain'] = None

# Prompt
if 'template' not in st.session_state:
    st.session_state['template'] = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
{context}
Question: {question}
Helpful Answer:"""

if 'QA_CHAIN_PROMPT' not in st.session_state:
    st.session_state['QA_CHAIN_PROMPT'] = PromptTemplate(
        input_variables=["context", "question"],
        template=st.session_state['template'],
    )

if 'llm' not in st.session_state:
    # st.session_state['llm'] = Ollama(base_url='https://ollama.api', model="llama3.1",
    #                                  callback_manager=CallbackManager([MyHandler(), StreamingStdOutCallbackHandler()]))
    st.session_state['llm'] = Ollama(model="llama3.1",
                                     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))


def inference(query):
    if st.session_state['qa_chain'] is None:
        st.warning("not initialized")
        return
    if query.strip() == "":
        st.warning("empty query")
        return

    st.info(st.session_state['qa_chain']({"query": query})['result'])


def download(url):
    if not url:
        return "Document URL is empty"

    st.info('Downloading...')

    loader = OnlinePDFLoader(url)
    data = loader.load()

    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    with SuppressStdout():
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

    st.session_state['qa_chain'] = RetrievalQA.from_chain_type(
        st.session_state['llm'],
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": st.session_state['QA_CHAIN_PROMPT']},
    )

    return st.info('Done')


url = st.text_area("URL:", "https://pdfobject.com/pdf/sample.pdf")
if st.button("Download"):
    download(url)

user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        # def generator(query):
        #     for token in st.session_state['qa_chain'].stream(query):
        #         yield token

        response = st.write_stream(inference(user_query))
        # response = st.write_stream(generator(user_query))
