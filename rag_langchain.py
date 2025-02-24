import os
from langchain.llms import HuggingFaceHub
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from secretkey import huggingface_api_key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_key

class StrOutputParser:
    def __call__(self, text: str) -> str:
        return text.strip()

def deduplicate_context(text: str) -> str:
    lines = text.splitlines()
    unique_lines = list(dict.fromkeys([line.strip() for line in lines if line.strip()]))
    return "\n".join(unique_lines)

def format_docs(docs):
    raw_text = "\n\n".join(doc.page_content for doc in docs)
    return deduplicate_context(raw_text)

def load_docs(url):
    loader = WebBaseLoader(url)
    return loader.load()

def get_text_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(docs)

def get_vector_store(chunks):
    vectorstore = Chroma.from_documents(documents=chunks, embedding=HuggingFaceEmbeddings())
    return vectorstore.as_retriever()

def get_rag_chain(retriever):
    llm = HuggingFaceHub(
        repo_id="tiiuae/falcon-7b-instruct",  # Free model
        model_kwargs={"temperature": 0.7, "max_length": 512}
    )
    template = """{context}

"""

    prompt = PromptTemplate.from_template(template)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def gen_answer(url, question):
    docs = load_docs(url)
    chunks = get_text_chunks(docs)
    retriever = get_vector_store(chunks)
    rag_chain = get_rag_chain(retriever)
    return rag_chain.invoke(question)