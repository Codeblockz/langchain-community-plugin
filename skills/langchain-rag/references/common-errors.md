# Common RAG Errors

## Table of Contents
- [Embedding Errors](#embedding-errors)
- [Vector Store Errors](#vector-store-errors)
- [Retrieval Errors](#retrieval-errors)
- [Document Loading Errors](#document-loading-errors)
- [Chain Errors](#chain-errors)

## Embedding Errors

### Dimension Mismatch

```
Error: ValueError: shapes (1536,) and (768,) not aligned
```

**Cause**: Embedding model dimension doesn't match vector store index.

```python
# WRONG - dimension mismatch
embeddings = OpenAIEmbeddings()  # 1536 dimensions
index = faiss.IndexFlatL2(768)   # Wrong!

# CORRECT - match dimensions
embedding_dim = len(embeddings.embed_query("test"))
index = faiss.IndexFlatL2(embedding_dim)  # 1536
```

### Rate Limiting

```
Error: openai.RateLimitError: Rate limit exceeded
```

**Cause**: Too many embedding requests.

```python
# SOLUTION - batch with delays
from langchain_openai import OpenAIEmbeddings
import time

embeddings = OpenAIEmbeddings(
    chunk_size=100,  # Reduce batch size
)

# Or add delays
def embed_with_retry(texts, embeddings, delay=1):
    results = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i+100]
        results.extend(embeddings.embed_documents(batch))
        time.sleep(delay)
    return results
```

### Empty Text

```
Error: ValueError: Cannot embed empty string
```

**Cause**: Document has no content after splitting.

```python
# SOLUTION - filter empty documents
chunks = splitter.split_documents(docs)
chunks = [c for c in chunks if c.page_content.strip()]
```

## Vector Store Errors

### Collection Not Found

```
Error: CollectionNotFoundError: Collection 'my_docs' not found
```

**Cause**: Trying to load non-existent collection.

```python
# SOLUTION - create if not exists
from langchain_chroma import Chroma

vectorstore = Chroma(
    collection_name="my_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)
# Collection auto-created if not exists
```

### FAISS Deserialization

```
Error: ValueError: allow_dangerous_deserialization must be True
```

**Cause**: Security check when loading FAISS index.

```python
# SOLUTION - explicitly allow (only for trusted files)
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
```

### Connection Failed

```
Error: ConnectionRefusedError: [Errno 111] Connection refused
```

**Cause**: Vector store server not running.

```python
# SOLUTION - verify server is running
# For Chroma: chroma run --path /db_path
# For Qdrant: docker run -p 6333:6333 qdrant/qdrant
# For Weaviate: docker-compose up -d

# Add retry logic
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def get_vectorstore():
    return Chroma(
        client=chromadb.HttpClient(host="localhost", port=8000),
        collection_name="my_docs",
        embedding_function=embeddings,
    )
```

## Retrieval Errors

### No Results

```python
# Returns empty list
docs = retriever.invoke("query")
print(len(docs))  # 0
```

**Causes & Solutions**:

1. **Index is empty**
```python
# Check document count
print(f"Documents in store: {vectorstore._collection.count()}")  # Chroma
```

2. **Query doesn't match content**
```python
# Try broader query or use multi-query retriever
from langchain.retrievers.multi_query import MultiQueryRetriever
retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
```

3. **Score threshold too high**
```python
# Lower threshold
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.3}  # Lower from 0.5
)
```

### Poor Relevance

**Cause**: Chunks too large, embeddings not matching query style.

```python
# SOLUTION 1 - smaller chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,   # Smaller
    chunk_overlap=100,
)

# SOLUTION 2 - use hypothetical document embeddings (HyDE)
from langchain.chains import HypotheticalDocumentEmbedder

hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
    llm=llm,
    base_embeddings=embeddings,
    prompt=...,
)

# SOLUTION 3 - add re-ranking
from langchain_cohere import CohereRerank
compressor = CohereRerank(model="rerank-english-v3.0", top_n=4)
```

### Duplicate Results

**Cause**: Same content in multiple chunks.

```python
# SOLUTION - use MMR or filter
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5}
)

# Or deduplicate manually
seen = set()
unique_docs = []
for doc in docs:
    content_hash = hash(doc.page_content[:100])
    if content_hash not in seen:
        seen.add(content_hash)
        unique_docs.append(doc)
```

## Document Loading Errors

### PDF Parsing Failed

```
Error: PdfReadError: Could not read malformed PDF file
```

**Solution**: Try different loader.

```python
# Option 1: PyMuPDF (better for complex PDFs)
from langchain_community.document_loaders import PyMuPDFLoader
loader = PyMuPDFLoader("document.pdf")

# Option 2: Unstructured (handles most formats)
from langchain_community.document_loaders import UnstructuredPDFLoader
loader = UnstructuredPDFLoader("document.pdf", mode="elements")

# Option 3: OCR for scanned PDFs
# pip install pytesseract pdf2image
from langchain_community.document_loaders import UnstructuredPDFLoader
loader = UnstructuredPDFLoader("scanned.pdf", strategy="ocr_only")
```

### Encoding Error

```
Error: UnicodeDecodeError: 'utf-8' codec can't decode byte
```

```python
# SOLUTION - specify encoding
from langchain_community.document_loaders import TextLoader

loader = TextLoader("document.txt", encoding="latin-1")
# Or try: encoding="cp1252", encoding="iso-8859-1"
```

### Web Page Not Loading

```
Error: requests.exceptions.SSLError
```

```python
# SOLUTION - disable SSL verification (not recommended for production)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Or use different loader
from langchain_community.document_loaders import SeleniumURLLoader
loader = SeleniumURLLoader(urls=["https://example.com"])
```

## Chain Errors

### Context Too Long

```
Error: InvalidRequestError: This model's maximum context length is 8192 tokens
```

```python
# SOLUTION 1 - retrieve fewer docs
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# SOLUTION 2 - use smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500)

# SOLUTION 3 - compress context
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever,
)
```

### Empty Context

```python
# RAG chain returns: "I don't have enough information"
```

**Cause**: Retriever returned no documents.

```python
# SOLUTION - handle empty results
def format_docs(docs):
    if not docs:
        return "No relevant documents found."
    return "\n\n".join(doc.page_content for doc in docs)

# Or check before generating
docs = retriever.invoke(query)
if not docs:
    return "I couldn't find any relevant information."
```

## Debug Checklist

1. **Verify documents loaded**: `print(len(docs))`
2. **Check chunks created**: `print(len(chunks))`
3. **Confirm embeddings work**: `embeddings.embed_query("test")`
4. **Test vector store**: `vectorstore.similarity_search("test")`
5. **Inspect retriever output**: `print(retriever.invoke("query"))`
6. **Check metadata**: `print(docs[0].metadata)`

## Performance Tips

1. **Batch document adds**: Add in groups of 100-500
2. **Use async for web loaders**: `AsyncHtmlLoader`
3. **Cache embeddings**: Avoid re-embedding same content
4. **Index metadata fields**: Enable filtering in vector store
5. **Use appropriate chunk sizes**: Smaller for Q&A, larger for summarization
