---
name: langchain-rag
description: Build RAG (Retrieval-Augmented Generation) pipelines with LangChain. Use when loading documents, chunking text, setting up vector stores, creating retrievers, or building question-answering systems. Covers document loaders, text splitters, embeddings, and all major vector stores.
---

# LangChain RAG Builder

## Quick Decision: Which Vector Store?

| Use Case | Vector Store | Notes |
|----------|-------------|-------|
| **Quick prototyping** | `InMemoryVectorStore` | No setup, data lost on restart |
| **Local development** | `FAISS` or `Chroma` | File-based persistence |
| **Production (managed)** | `Pinecone` or `Qdrant` | Fully managed, scalable |
| **Production (self-hosted)** | `pgvector` or `Weaviate` | Use existing Postgres or K8s |

## RAG Pipeline Quick Start

```python
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Load documents
loader = WebBaseLoader("https://example.com/docs")
docs = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Characters per chunk
    chunk_overlap=200,    # Overlap between chunks
    add_start_index=True, # Track position in original doc
)
chunks = splitter.split_documents(docs)

# 3. Create vector store with embeddings
embeddings = OpenAIEmbeddings()
vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)

# 4. Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 5. Build RAG chain
prompt = ChatPromptTemplate.from_template("""
Answer based only on the context. If unsure, say "I don't know."

Context: {context}
Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

llm = ChatOpenAI(model="gpt-4o")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. Query
answer = rag_chain.invoke("What is this document about?")
```

## Critical Rules

1. **Match embedding dimensions** - Vector store dimension must match your embedding model
2. **Preserve metadata** - Use `add_start_index=True` to track chunk origins
3. **Tune chunk size** - Start with 1000 chars, adjust based on retrieval quality
4. **Use overlap** - 10-20% overlap prevents losing context at boundaries
5. **Handle empty results** - Check if retriever returns docs before generating

## Common Gotchas

### Embedding dimension mismatch
```python
# WRONG - dimension mismatch will fail
embeddings = OpenAIEmbeddings()  # 1536 dimensions
index = faiss.IndexFlatL2(768)   # Wrong dimension!

# CORRECT - match dimensions
embedding_dim = len(embeddings.embed_query("test"))
index = faiss.IndexFlatL2(embedding_dim)
```

### Lost metadata on split
```python
# WRONG - metadata lost
chunks = splitter.split_text(doc.page_content)

# CORRECT - metadata preserved
chunks = splitter.split_documents([doc])
```

### No results handling
```python
# WRONG - crashes on empty results
docs = retriever.invoke(query)
context = format_docs(docs)  # Fails if empty

# CORRECT - handle empty
docs = retriever.invoke(query)
if not docs:
    return "No relevant documents found"
context = format_docs(docs)
```

## Chunk Size Selection

| Content Type | Recommended Size | Overlap |
|-------------|------------------|---------|
| Technical docs | 500-1000 chars | 100-200 |
| Legal/dense text | 300-500 chars | 50-100 |
| Conversational | 1000-2000 chars | 200-400 |
| Code | 500-1000 chars | 100-200 |

## Reference Documentation

Read these for detailed patterns:

- **[document-loaders.md](references/document-loaders.md)** - PDF, web, CSV, JSON loaders
- **[text-splitters.md](references/text-splitters.md)** - Chunking strategies
- **[vector-stores.md](references/vector-stores.md)** - FAISS, Chroma, pgvector, Pinecone, Qdrant, Weaviate
- **[retriever-patterns.md](references/retriever-patterns.md)** - Hybrid search, re-ranking, multi-query
- **[common-errors.md](references/common-errors.md)** - Error codes and fixes
