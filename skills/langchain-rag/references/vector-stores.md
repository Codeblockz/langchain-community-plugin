# Vector Stores

## Table of Contents
- [Quick Comparison](#quick-comparison)
- [InMemoryVectorStore](#inmemoryvectorstore)
- [FAISS](#faiss)
- [Chroma](#chroma)
- [pgvector](#pgvector)
- [Pinecone](#pinecone)
- [Qdrant](#qdrant)
- [Weaviate](#weaviate)

## Quick Comparison

| Store | Persistence | Scalability | Setup | Best For |
|-------|------------|-------------|-------|----------|
| InMemory | None | Low | None | Prototyping |
| FAISS | File | Medium | `pip install faiss-cpu` | Local dev |
| Chroma | File/Server | Medium | `pip install chromadb` | Local dev |
| pgvector | PostgreSQL | High | Postgres extension | Existing Postgres |
| Pinecone | Cloud | Very High | API key | Production (managed) |
| Qdrant | Cloud/Self | Very High | Docker or cloud | Production |
| Weaviate | Cloud/Self | Very High | Docker or cloud | Production |

## InMemoryVectorStore

Zero setup, data lost on restart.

```python
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = InMemoryVectorStore(embeddings)

# Add documents
vectorstore.add_documents(docs)

# Or create from documents
vectorstore = InMemoryVectorStore.from_documents(docs, embeddings)

# Search
results = vectorstore.similarity_search("query", k=4)
```

## FAISS

Facebook's efficient similarity search. File-based persistence.

```python
# pip install faiss-cpu  (or faiss-gpu for GPU)

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Create from documents
vectorstore = FAISS.from_documents(docs, embeddings)

# Save to disk
vectorstore.save_local("faiss_index")

# Load from disk
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True  # Required for loading
)

# Search with scores
results = vectorstore.similarity_search_with_score("query", k=4)
for doc, score in results:
    print(f"Score: {score:.4f} - {doc.page_content[:50]}")
```

### FAISS with Custom Index

```python
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# Get embedding dimension
embedding_dim = len(embeddings.embed_query("test"))

# Create FAISS index
index = faiss.IndexFlatL2(embedding_dim)  # L2 distance
# Or: faiss.IndexFlatIP(embedding_dim)    # Inner product (cosine)

vectorstore = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
```

## Chroma

Easy setup with persistence. Good for development.

```python
# pip install chromadb

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Ephemeral (in-memory)
vectorstore = Chroma(
    collection_name="my_collection",
    embedding_function=embeddings,
)

# Persistent (file-based)
vectorstore = Chroma(
    collection_name="my_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

# Add documents
vectorstore.add_documents(docs)

# Search with filter
results = vectorstore.similarity_search(
    "query",
    k=4,
    filter={"source": "important.pdf"}
)
```

### Chroma Server Mode

```python
# Start server: chroma run --path /db_path

import chromadb
from langchain_chroma import Chroma

client = chromadb.HttpClient(host="localhost", port=8000)

vectorstore = Chroma(
    client=client,
    collection_name="my_collection",
    embedding_function=embeddings,
)
```

## pgvector

PostgreSQL extension. Use existing database infrastructure.

```python
# pip install langchain-postgres

from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

CONNECTION_STRING = "postgresql+psycopg://user:pass@localhost:5432/dbname"

vectorstore = PGVector(
    embeddings=embeddings,
    collection_name="my_docs",
    connection=CONNECTION_STRING,
)

# Add documents
vectorstore.add_documents(docs)

# Search
results = vectorstore.similarity_search("query", k=4)
```

### pgvector with SQLAlchemy Engine

```python
from langchain_postgres import PGEngine, PGVectorStore

engine = PGEngine.from_connection_string(
    url="postgresql+psycopg://user:pass@localhost:5432/dbname"
)

vectorstore = PGVectorStore.create_sync(
    engine=engine,
    table_name="documents",
    embedding_service=embeddings,
)
```

## Pinecone

Fully managed, highly scalable. Production-ready.

```python
# pip install langchain-pinecone pinecone-client

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

# Initialize Pinecone
pc = Pinecone(api_key="your-api-key")
index = pc.Index("your-index-name")

embeddings = OpenAIEmbeddings()

# Create vector store
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
)

# Add documents with namespace
vectorstore.add_documents(docs, namespace="production")

# Search with namespace
results = vectorstore.similarity_search(
    "query",
    k=4,
    namespace="production"
)
```

### Pinecone with Metadata Filtering

```python
# Search with metadata filter
results = vectorstore.similarity_search(
    "query",
    k=4,
    filter={
        "source": {"$eq": "docs.pdf"},
        "page": {"$gte": 10}
    }
)
```

## Qdrant

High-performance, supports hybrid search.

```python
# pip install qdrant-client langchain-qdrant

from langchain_qdrant import Qdrant
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# In-memory
vectorstore = Qdrant.from_documents(
    docs,
    embeddings,
    location=":memory:",
    collection_name="my_docs",
)

# Persistent (local)
vectorstore = Qdrant.from_documents(
    docs,
    embeddings,
    path="./qdrant_db",
    collection_name="my_docs",
)

# Cloud
from qdrant_client import QdrantClient

client = QdrantClient(
    url="https://your-cluster.qdrant.io",
    api_key="your-api-key",
)

vectorstore = Qdrant(
    client=client,
    collection_name="my_docs",
    embeddings=embeddings,
)
```

## Weaviate

GraphQL-based, supports hybrid search.

```python
# pip install langchain-weaviate weaviate-client

from langchain_weaviate import WeaviateVectorStore
from langchain_openai import OpenAIEmbeddings
import weaviate

embeddings = OpenAIEmbeddings()

# Connect to Weaviate
client = weaviate.connect_to_local()  # Or connect_to_wcs() for cloud

vectorstore = WeaviateVectorStore(
    client=client,
    index_name="Documents",
    text_key="content",
    embedding=embeddings,
)

# Add documents
vectorstore.add_documents(docs)

# Search
results = vectorstore.similarity_search("query", k=4)

# Don't forget to close
client.close()
```

### Weaviate Cloud

```python
import weaviate
from weaviate.auth import Auth

client = weaviate.connect_to_wcs(
    cluster_url="https://your-cluster.weaviate.network",
    auth_credentials=Auth.api_key("your-api-key"),
)
```

## Common Operations

### Add Documents

```python
# All vector stores support these methods
vectorstore.add_documents(docs)
vectorstore.add_texts(texts, metadatas=metadatas)
```

### Search Methods

```python
# Basic similarity search
results = vectorstore.similarity_search("query", k=4)

# With scores
results = vectorstore.similarity_search_with_score("query", k=4)

# With relevance scores (normalized 0-1)
results = vectorstore.similarity_search_with_relevance_scores("query", k=4)

# MMR (Maximal Marginal Relevance) - diverse results
results = vectorstore.max_marginal_relevance_search("query", k=4, fetch_k=20)
```

### Convert to Retriever

```python
# Basic retriever
retriever = vectorstore.as_retriever()

# With search parameters
retriever = vectorstore.as_retriever(
    search_type="similarity",  # or "mmr", "similarity_score_threshold"
    search_kwargs={
        "k": 4,
        "score_threshold": 0.5,  # For similarity_score_threshold
        "fetch_k": 20,           # For mmr
        "lambda_mult": 0.5,      # For mmr (diversity)
    }
)
```

## Best Practices

1. **Match dimensions** - Embedding model dimension must match index
2. **Use namespaces** - Organize data by environment/tenant
3. **Index metadata** - Enable filtering on common fields
4. **Monitor costs** - Cloud services charge per operation
5. **Batch operations** - Add documents in batches for efficiency
6. **Handle failures** - Implement retry logic for cloud services
