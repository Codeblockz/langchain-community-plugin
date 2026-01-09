# Document Loaders

## Table of Contents
- [Web Loaders](#web-loaders)
- [PDF Loaders](#pdf-loaders)
- [Structured Data](#structured-data)
- [Text Files](#text-files)
- [Custom Loaders](#custom-loaders)

## Web Loaders

### WebBaseLoader (Simple HTML)

```python
from langchain_community.document_loaders import WebBaseLoader

# Single URL
loader = WebBaseLoader("https://example.com/page")
docs = loader.load()

# Multiple URLs
loader = WebBaseLoader([
    "https://example.com/page1",
    "https://example.com/page2",
])
docs = loader.load()
```

### AsyncHtmlLoader (Concurrent)

```python
from langchain_community.document_loaders import AsyncHtmlLoader

loader = AsyncHtmlLoader([
    "https://example.com/page1",
    "https://example.com/page2",
])
docs = await loader.aload()
```

### RecursiveUrlLoader (Crawl Site)

```python
from langchain_community.document_loaders import RecursiveUrlLoader

loader = RecursiveUrlLoader(
    url="https://docs.example.com",
    max_depth=2,  # How deep to crawl
    extractor=lambda x: x.text,  # Custom text extraction
)
docs = loader.load()
```

## PDF Loaders

### PyPDFLoader (Recommended)

```python
from langchain_community.document_loaders import PyPDFLoader

# pip install pypdf

loader = PyPDFLoader("document.pdf")
pages = loader.load()  # Each page is a Document

# Access page metadata
print(pages[0].metadata)  # {'source': 'document.pdf', 'page': 0}
```

### PyMuPDFLoader (Better Formatting)

```python
from langchain_community.document_loaders import PyMuPDFLoader

# pip install pymupdf

loader = PyMuPDFLoader("document.pdf")
pages = loader.load()  # Preserves more formatting
```

### UnstructuredPDFLoader (Complex PDFs)

```python
from langchain_community.document_loaders import UnstructuredPDFLoader

# pip install unstructured

loader = UnstructuredPDFLoader(
    "complex_document.pdf",
    mode="elements",  # Preserves structure
)
docs = loader.load()
```

## Structured Data

### CSVLoader

```python
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(
    file_path="data.csv",
    source_column="id",  # Column to use as source metadata
)
docs = loader.load()

# Each row becomes a Document
print(docs[0].page_content)  # "column1: value1\ncolumn2: value2"
```

### JSONLoader

```python
from langchain_community.document_loaders import JSONLoader

# Simple JSON array
loader = JSONLoader(
    file_path="data.json",
    jq_schema=".[]",  # JSONPath to iterate
    text_content=False,
)
docs = loader.load()

# Nested JSON
loader = JSONLoader(
    file_path="data.json",
    jq_schema=".results[].content",
    metadata_func=lambda record, metadata: {**metadata, "id": record.get("id")},
)
```

### DataFrameLoader

```python
from langchain_community.document_loaders import DataFrameLoader
import pandas as pd

df = pd.read_csv("data.csv")
loader = DataFrameLoader(
    df,
    page_content_column="text",  # Column with main content
)
docs = loader.load()
```

## Text Files

### TextLoader

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("document.txt", encoding="utf-8")
docs = loader.load()
```

### DirectoryLoader

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Load all .txt files from directory
loader = DirectoryLoader(
    path="./documents",
    glob="**/*.txt",
    loader_cls=TextLoader,
    show_progress=True,
)
docs = loader.load()

# Load multiple file types
from langchain_community.document_loaders import PyPDFLoader

loader = DirectoryLoader(
    path="./documents",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
)
```

### UnstructuredFileLoader (Auto-detect)

```python
from langchain_community.document_loaders import UnstructuredFileLoader

# Automatically detects file type
loader = UnstructuredFileLoader("document.docx")
docs = loader.load()
```

## Custom Loaders

### Create Custom Loader

```python
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from typing import Iterator

class CustomAPILoader(BaseLoader):
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load documents one at a time."""
        import requests

        response = requests.get(
            self.api_url,
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        data = response.json()

        for item in data["results"]:
            yield Document(
                page_content=item["content"],
                metadata={
                    "source": self.api_url,
                    "id": item["id"],
                }
            )

# Usage
loader = CustomAPILoader("https://api.example.com/docs", "key")
docs = loader.load()  # Or use lazy_load() for streaming
```

## Loader Selection Guide

| File Type | Recommended Loader | Install |
|-----------|-------------------|---------|
| HTML/Web | `WebBaseLoader` | Built-in |
| PDF (simple) | `PyPDFLoader` | `pypdf` |
| PDF (complex) | `UnstructuredPDFLoader` | `unstructured` |
| CSV | `CSVLoader` | Built-in |
| JSON | `JSONLoader` | Built-in |
| Word (.docx) | `UnstructuredFileLoader` | `unstructured` |
| Markdown | `UnstructuredMarkdownLoader` | `unstructured` |
| Directory | `DirectoryLoader` | Built-in |

## Best Practices

1. **Always check metadata** - Verify source tracking is preserved
2. **Use lazy_load for large files** - Prevents memory issues
3. **Handle encoding** - Specify `encoding="utf-8"` for text files
4. **Add custom metadata** - Include timestamps, sources, versions
5. **Validate content** - Check documents aren't empty after loading
