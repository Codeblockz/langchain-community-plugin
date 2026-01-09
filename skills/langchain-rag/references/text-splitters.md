# Text Splitters

## Table of Contents
- [Overview](#overview)
- [RecursiveCharacterTextSplitter](#recursivecharactertextsplitter)
- [Token-Based Splitting](#token-based-splitting)
- [Semantic Splitting](#semantic-splitting)
- [Code Splitting](#code-splitting)
- [Custom Splitters](#custom-splitters)

## Overview

Text splitters break documents into chunks for embedding. Key parameters:

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `chunk_size` | Max characters/tokens per chunk | 500-2000 |
| `chunk_overlap` | Overlap between chunks | 10-20% of chunk_size |
| `add_start_index` | Track position in original doc | `True` recommended |

## RecursiveCharacterTextSplitter

**Recommended for most use cases.** Recursively splits on separators (paragraphs → sentences → words).

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Basic usage
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
    separators=["\n\n", "\n", " ", ""],  # Default
)

# Split text
chunks = splitter.split_text(text)

# Split documents (preserves metadata)
doc_chunks = splitter.split_documents(documents)
```

### Custom Separators

```python
# For Markdown
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=[
        "\n## ",     # H2 headers
        "\n### ",    # H3 headers
        "\n\n",      # Paragraphs
        "\n",        # Lines
        " ",         # Words
        "",          # Characters
    ],
)

# For HTML
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["</div>", "</p>", "<br>", "\n\n", "\n", " ", ""],
)
```

## Token-Based Splitting

Better for LLM context limits since tokens != characters.

### Using tiktoken (OpenAI)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Token-based (recommended for OpenAI models)
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",  # GPT-4, GPT-3.5-turbo
    chunk_size=500,               # Tokens, not characters
    chunk_overlap=50,
)
chunks = splitter.split_documents(docs)
```

### Using CharacterTextSplitter with Tokens

```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=500,
    chunk_overlap=0,
)
```

## Semantic Splitting

Split based on meaning, not just characters.

### SemanticChunker (Experimental)

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

# Requires embeddings to detect semantic boundaries
splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95,
)
chunks = splitter.split_documents(docs)
```

### SentenceTransformersTokenTextSplitter

```python
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=50,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
```

## Code Splitting

Language-aware splitting that respects code structure.

```python
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language,
)

# Python
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1000,
    chunk_overlap=200,
)

# JavaScript
js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS,
    chunk_size=1000,
    chunk_overlap=200,
)

# Supported languages
# Language.PYTHON, Language.JS, Language.TS, Language.GO,
# Language.RUST, Language.JAVA, Language.CPP, Language.PHP,
# Language.RUBY, Language.MARKDOWN, Language.HTML, Language.JSON
```

## Custom Splitters

### By Document Type

```python
def get_splitter_for_doc(doc):
    """Select splitter based on document type."""
    source = doc.metadata.get("source", "")

    if source.endswith(".py"):
        return RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=1000,
            chunk_overlap=200,
        )
    elif source.endswith(".md"):
        return RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
        )
    else:
        return RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
```

### With Metadata Enrichment

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def split_with_metadata(docs, base_splitter):
    """Split documents and enrich metadata."""
    all_chunks = []

    for doc in docs:
        chunks = base_splitter.split_documents([doc])

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
            chunk.metadata["char_count"] = len(chunk.page_content)
            all_chunks.append(chunk)

    return all_chunks

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
enriched_chunks = split_with_metadata(docs, splitter)
```

## Chunk Size Guidelines

| Model | Max Context | Recommended Chunk Size |
|-------|-------------|----------------------|
| GPT-4o | 128K tokens | 500-1000 tokens |
| GPT-4 | 8K tokens | 300-500 tokens |
| Claude 3.5 | 200K tokens | 500-1000 tokens |
| Local (Llama) | 4K-8K tokens | 200-400 tokens |

## Best Practices

1. **Start conservative** - Begin with smaller chunks, increase if needed
2. **Measure retrieval quality** - Test with real queries
3. **Match to model** - Use token-based for LLMs
4. **Preserve structure** - Use language-aware splitters for code
5. **Track origins** - Always use `add_start_index=True`
6. **Consider overlap** - 10-20% helps maintain context
