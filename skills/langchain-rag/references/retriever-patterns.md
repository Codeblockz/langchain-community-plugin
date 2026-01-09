# Retriever Patterns

## Table of Contents
- [Basic Retriever](#basic-retriever)
- [Multi-Query Retriever](#multi-query-retriever)
- [Contextual Compression](#contextual-compression)
- [Ensemble/Hybrid Search](#ensemblehybrid-search)
- [Parent Document Retriever](#parent-document-retriever)
- [Self-Query Retriever](#self-query-retriever)
- [Time-Weighted Retriever](#time-weighted-retriever)

## Basic Retriever

Convert any vector store to a retriever.

```python
from langchain_core.vectorstores import InMemoryVectorStore

# Basic retriever
retriever = vectorstore.as_retriever()

# With parameters
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# Use in chain
docs = retriever.invoke("What is RAG?")
```

### Search Types

```python
# Similarity (default)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# Similarity with score threshold
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.5, "k": 10}
)

# MMR (diverse results)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 20,      # Fetch more, then select diverse
        "lambda_mult": 0.5,  # 0=max diversity, 1=max relevance
    }
)
```

## Multi-Query Retriever

Generate multiple query variations for better recall.

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm,
)

# Generates multiple query variations, retrieves for each, deduplicates
docs = multi_retriever.invoke("What are the benefits of RAG?")
```

### Custom Query Generation

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["question"],
    template="""Generate 3 different versions of this question for document retrieval.

Original: {question}

Provide each variation on a new line."""
)

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm,
    prompt=prompt,
)
```

## Contextual Compression

Filter and compress retrieved documents for relevance.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Compressor extracts relevant portions
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
)

# Returns compressed, relevant excerpts
docs = compression_retriever.invoke("What is the main idea?")
```

### With Embeddings Filter

```python
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Filter by embedding similarity
embeddings_filter = EmbeddingsFilter(
    embeddings=embeddings,
    similarity_threshold=0.76,
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
)
```

### Pipeline Compressor

```python
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter

# Combine multiple compressors
pipeline = DocumentCompressorPipeline(
    transformers=[
        EmbeddingsRedundantFilter(embeddings=embeddings),  # Remove duplicates
        EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76),
    ]
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20}),
)
```

## Ensemble/Hybrid Search

Combine multiple retrievers or search methods.

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Sparse retriever (keyword-based)
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 4

# Dense retriever (embedding-based)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Combine with weights
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5],  # Equal weight to keyword and semantic
)

docs = ensemble_retriever.invoke("What is machine learning?")
```

### With Re-ranking

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# Re-rank with Cohere
compressor = CohereRerank(
    model="rerank-english-v3.0",
    top_n=4,
)

rerank_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=ensemble_retriever,
)
```

## Parent Document Retriever

Retrieve small chunks, return parent documents.

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Small chunks for retrieval
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

# Large chunks for context
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

# Store for parent documents
store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# Add documents (splits into children, stores parents)
retriever.add_documents(docs)

# Retrieves children, returns parents
parent_docs = retriever.invoke("query")
```

## Self-Query Retriever

Automatically extracts filters from natural language.

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain_openai import ChatOpenAI

# Define searchable metadata
metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The source document name",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="The page number",
        type="integer",
    ),
    AttributeInfo(
        name="date",
        description="The document date",
        type="string",
    ),
]

llm = ChatOpenAI(model="gpt-4o", temperature=0)

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="Technical documentation",
    metadata_field_info=metadata_field_info,
)

# Automatically extracts filter: page > 10
docs = retriever.invoke("What's on pages after 10?")
```

## Time-Weighted Retriever

Prefer recent documents.

```python
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from datetime import datetime

retriever = TimeWeightedVectorStoreRetriever(
    vectorstore=vectorstore,
    decay_rate=0.01,  # How quickly to decay relevance
    k=4,
)

# Add document with timestamp
retriever.add_documents([
    Document(
        page_content="Recent news",
        metadata={"last_accessed_at": datetime.now()}
    )
])

# Balances relevance with recency
docs = retriever.invoke("Latest updates")
```

## Custom Retriever

Create your own retriever logic.

```python
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List

class CustomRetriever(BaseRetriever):
    vectorstore: Any
    k: int = 4
    min_score: float = 0.5

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Custom retrieval logic."""
        # Get results with scores
        results = self.vectorstore.similarity_search_with_score(query, k=self.k * 2)

        # Filter by score
        filtered = [
            doc for doc, score in results
            if score >= self.min_score
        ]

        return filtered[:self.k]

# Usage
retriever = CustomRetriever(vectorstore=vectorstore, k=4, min_score=0.6)
```

## Best Practices

1. **Start simple** - Basic retriever first, add complexity as needed
2. **Measure recall** - Track if relevant docs are being retrieved
3. **Use hybrid for production** - Combine keyword + semantic
4. **Re-rank for precision** - Add re-ranking for better top results
5. **Cache expensive operations** - Multi-query and compression add latency
6. **Monitor token usage** - Compression uses LLM calls
