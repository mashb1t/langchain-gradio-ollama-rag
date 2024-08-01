import os
import sys
import threading
import time
from typing import Any, Dict

import gradio as gr
from langchain import PromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
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


qa_chain: BaseRetrievalQA | None = None

# Prompt
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

last_tokens = []


class MyHandler(AsyncCallbackHandler):
    def on_llm_new_token(
            self,
            token: str,
            **kwargs: Any,
    ) -> None:
        last_tokens.append(token)

    def on_llm_end(
            self,
            outputs: Dict[str, Any],
            **kwargs: Any,
    ) -> None:
        last_tokens.append('$$end$$')


# llm = Ollama(base_url='https://ollama.api', model="llama3.1",
#              callback_manager=CallbackManager([MyHandler(), StreamingStdOutCallbackHandler()]))
llm = Ollama(model="llama3.1", callback_manager=CallbackManager([MyHandler(), StreamingStdOutCallbackHandler()]))


def inference(query):
    global last_tokens
    last_tokens = []
    if qa_chain is None:
        return "not initialized"
    if query.strip() == "":
        return "empty query"

    thr = threading.Thread(target=qa_chain, args=({"query": query},))
    thr.start()

    response = ''

    while True:
        time.sleep(0.001)
        if len(last_tokens) == 0:
            continue
        while len(last_tokens) > 0:
            token = last_tokens.pop(0)
            if token == '$$end$$':
                return response
            response += token
            yield response


def download(url):
    global qa_chain

    if not url:
        return "Document URL is empty"
    loader = OnlinePDFLoader(url)
    data = loader.load()

    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    with SuppressStdout():
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )


with gr.Blocks().queue() as demo:
    # https://pdfobject.com/pdf/sample.pdf
    # https://d18rn0p25nwr6d.cloudfront.net/CIK-0001813756/975b3e9b-268e-4798-a9e4-2a9a7c92dc10.pdf
    url = gr.Textbox(label="Document URL")
    download_btn = gr.Button("Download")
    query = gr.Textbox(label="Query")

    generate_btn = gr.Button("Generate")
    output = gr.Textbox(label="Output")

    download_btn.click(fn=download, inputs=url, outputs=output)
    generate_btn.click(fn=inference, inputs=query, outputs=output)

demo.launch()
