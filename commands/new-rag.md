---
name: new-rag
description: Scaffold a new RAG (Retrieval-Augmented Generation) pipeline with best practices
allowed-tools:
  - Read
  - Write
  - AskUserQuestion
argument-hint: "[filename]"
---

# New RAG Pipeline Command

Create a new RAG pipeline file with best practices baked in.

## Workflow

1. **Ask the user** which vector store they want:
   - `InMemory` - Quick prototyping, no persistence
   - `FAISS` - Local, file-based persistence
   - `Chroma` - Local with server option
   - `pgvector` - PostgreSQL-based
   - `Pinecone` - Managed cloud service

2. **Get filename** from argument or ask user (default: `rag_pipeline.py`)

3. **Generate the RAG file** using the appropriate template below

4. **Inform user** about next steps (install dependencies, configure embeddings)

## Templates

### InMemory Template

```python
"""
RAG Pipeline using InMemoryVectorStore

Install: pip install langchain langchain-openai
"""
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


# 1. Load documents
def load_documents(urls: list[str]):
    """Load documents from URLs."""
    docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        docs.extend(loader.load())
    return docs


# 2. Split into chunks
def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(docs)


# 3. Create vector store
def create_vectorstore(chunks):
    """Create vector store from document chunks."""
    embeddings = OpenAIEmbeddings()
    vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)
    return vectorstore


# 4. Build RAG chain
def build_rag_chain(vectorstore):
    """Build the RAG chain."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context.
If you cannot answer based on the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:
""")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def main():
    # Example usage
    urls = [
        "https://example.com/page1",
        "https://example.com/page2",
    ]

    print("Loading documents...")
    docs = load_documents(urls)
    print(f"Loaded {len(docs)} documents")

    print("Splitting documents...")
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    print("Creating vector store...")
    vectorstore = create_vectorstore(chunks)

    print("Building RAG chain...")
    chain = build_rag_chain(vectorstore)

    # Query
    question = "What is this about?"
    print(f"\nQuestion: {question}")
    answer = chain.invoke(question)
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
```

### FAISS Template

```python
"""
RAG Pipeline using FAISS

Install: pip install langchain langchain-openai faiss-cpu
"""
from pathlib import Path
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


INDEX_PATH = "./faiss_index"


def load_documents(urls: list[str]):
    """Load documents from URLs."""
    docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        docs.extend(loader.load())
    return docs


def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(docs)


def get_or_create_vectorstore(chunks=None):
    """Load existing index or create new one."""
    embeddings = OpenAIEmbeddings()

    if Path(INDEX_PATH).exists() and chunks is None:
        print("Loading existing FAISS index...")
        return FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    if chunks is None:
        raise ValueError("No existing index and no chunks provided")

    print("Creating new FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)
    return vectorstore


def build_rag_chain(vectorstore):
    """Build the RAG chain."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context.
If you cannot answer based on the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:
""")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def main():
    # Set to True to rebuild index
    rebuild_index = True

    if rebuild_index:
        urls = ["https://example.com/page1"]
        docs = load_documents(urls)
        chunks = split_documents(docs)
        vectorstore = get_or_create_vectorstore(chunks)
    else:
        vectorstore = get_or_create_vectorstore()

    chain = build_rag_chain(vectorstore)

    question = "What is this about?"
    answer = chain.invoke(question)
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
```

### Chroma Template

```python
"""
RAG Pipeline using Chroma

Install: pip install langchain langchain-openai langchain-chroma
"""
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "my_documents"


def load_documents(urls: list[str]):
    """Load documents from URLs."""
    docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        docs.extend(loader.load())
    return docs


def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(docs)


def get_or_create_vectorstore(chunks=None):
    """Get existing collection or create new one."""
    embeddings = OpenAIEmbeddings()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )

    if chunks:
        print("Adding documents to Chroma...")
        vectorstore.add_documents(chunks)

    return vectorstore


def build_rag_chain(vectorstore):
    """Build the RAG chain."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context.
If you cannot answer based on the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:
""")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def main():
    # Set to True to add new documents
    add_documents = True

    if add_documents:
        urls = ["https://example.com/page1"]
        docs = load_documents(urls)
        chunks = split_documents(docs)
        vectorstore = get_or_create_vectorstore(chunks)
    else:
        vectorstore = get_or_create_vectorstore()

    chain = build_rag_chain(vectorstore)

    question = "What is this about?"
    answer = chain.invoke(question)
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
```

### pgvector Template

```python
"""
RAG Pipeline using pgvector (PostgreSQL)

Install: pip install langchain langchain-openai langchain-postgres psycopg[binary]
Requires: PostgreSQL with pgvector extension
"""
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


# Update with your connection string
CONNECTION_STRING = "postgresql+psycopg://user:password@localhost:5432/vectordb"
COLLECTION_NAME = "my_documents"


def load_documents(urls: list[str]):
    """Load documents from URLs."""
    docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        docs.extend(loader.load())
    return docs


def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(docs)


def get_vectorstore():
    """Get pgvector store."""
    embeddings = OpenAIEmbeddings()

    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
    )

    return vectorstore


def add_documents(vectorstore, chunks):
    """Add documents to vector store."""
    vectorstore.add_documents(chunks)


def build_rag_chain(vectorstore):
    """Build the RAG chain."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context.
If you cannot answer based on the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:
""")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def main():
    vectorstore = get_vectorstore()

    # Uncomment to add documents
    # urls = ["https://example.com/page1"]
    # docs = load_documents(urls)
    # chunks = split_documents(docs)
    # add_documents(vectorstore, chunks)

    chain = build_rag_chain(vectorstore)

    question = "What is this about?"
    answer = chain.invoke(question)
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
```

### Pinecone Template

```python
"""
RAG Pipeline using Pinecone

Install: pip install langchain langchain-openai langchain-pinecone pinecone-client
"""
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone


# Configure these
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "my-index"
NAMESPACE = "default"


def load_documents(urls: list[str]):
    """Load documents from URLs."""
    docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        docs.extend(loader.load())
    return docs


def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(docs)


def get_vectorstore():
    """Get Pinecone vector store."""
    embeddings = OpenAIEmbeddings()

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=NAMESPACE,
    )

    return vectorstore


def add_documents(vectorstore, chunks):
    """Add documents to Pinecone."""
    vectorstore.add_documents(chunks)


def build_rag_chain(vectorstore):
    """Build the RAG chain."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context.
If you cannot answer based on the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:
""")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def main():
    vectorstore = get_vectorstore()

    # Uncomment to add documents
    # urls = ["https://example.com/page1"]
    # docs = load_documents(urls)
    # chunks = split_documents(docs)
    # add_documents(vectorstore, chunks)

    chain = build_rag_chain(vectorstore)

    question = "What is this about?"
    answer = chain.invoke(question)
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
```

## After Generation

Tell the user:
1. Install dependencies listed in the docstring
2. Set environment variables (OPENAI_API_KEY, etc.)
3. Update the document URLs or loaders
4. Run with: `python <filename>`
