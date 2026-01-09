# Summarization Patterns

## Table of Contents
- [Strategy Selection](#strategy-selection)
- [Stuff Strategy](#stuff-strategy)
- [Map-Reduce Strategy](#map-reduce-strategy)
- [Refine Strategy](#refine-strategy)
- [Conversation Summarization](#conversation-summarization)

## Strategy Selection

| Strategy | Best For | Pros | Cons |
|----------|----------|------|------|
| **Stuff** | Small docs (<10k tokens) | Simple, single call | Limited by context window |
| **Map-Reduce** | Large docs, parallelizable | Fast, scalable | May lose cross-chunk context |
| **Refine** | Long docs needing coherence | Maintains context | Slow, sequential |

## Stuff Strategy

Fit all content into one prompt. Simplest approach.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("""
Summarize the following documents in a concise manner.
Focus on the key points and main ideas.

Documents:
{documents}

Summary:
""")

llm = ChatOpenAI(model="gpt-4o")

stuff_chain = prompt | llm | StrOutputParser()

# Combine all documents
combined = "\n\n---\n\n".join([doc.page_content for doc in docs])
summary = stuff_chain.invoke({"documents": combined})
```

### With Metadata

```python
prompt = ChatPromptTemplate.from_template("""
Summarize the following documents, noting their sources.

Documents:
{documents}

Provide a summary with source attribution.
""")

def format_docs_with_source(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        formatted.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

combined = format_docs_with_source(docs)
summary = stuff_chain.invoke({"documents": combined})
```

## Map-Reduce Strategy

Summarize chunks in parallel, then combine.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

# Map: Summarize each chunk
map_prompt = ChatPromptTemplate.from_template("""
Summarize the following section concisely:

{chunk}

Summary:
""")

map_chain = map_prompt | llm | StrOutputParser()

# Reduce: Combine summaries
reduce_prompt = ChatPromptTemplate.from_template("""
The following are summaries of different sections of a document.
Combine them into a single, coherent summary.

Section Summaries:
{summaries}

Combined Summary:
""")

reduce_chain = reduce_prompt | llm | StrOutputParser()

# Execute map-reduce
def map_reduce_summarize(docs, chunk_size=4000):
    # Split into chunks if needed
    chunks = [doc.page_content for doc in docs]

    # Map: Summarize each chunk (can parallelize)
    chunk_summaries = [
        map_chain.invoke({"chunk": chunk})
        for chunk in chunks
    ]

    # Reduce: Combine summaries
    final_summary = reduce_chain.invoke({
        "summaries": "\n\n---\n\n".join(chunk_summaries)
    })

    return final_summary

summary = map_reduce_summarize(docs)
```

### Parallel Map

```python
import asyncio

async def parallel_map_reduce(docs):
    # Map in parallel
    tasks = [
        map_chain.ainvoke({"chunk": doc.page_content})
        for doc in docs
    ]
    chunk_summaries = await asyncio.gather(*tasks)

    # Reduce
    final_summary = await reduce_chain.ainvoke({
        "summaries": "\n\n---\n\n".join(chunk_summaries)
    })

    return final_summary

summary = asyncio.run(parallel_map_reduce(docs))
```

### Hierarchical Map-Reduce

For very large documents, summarize in multiple rounds.

```python
def hierarchical_summarize(chunks, batch_size=5):
    current_level = chunks

    while len(current_level) > 1:
        next_level = []

        # Process in batches
        for i in range(0, len(current_level), batch_size):
            batch = current_level[i:i + batch_size]
            combined = "\n\n---\n\n".join(batch)
            summary = map_chain.invoke({"chunk": combined})
            next_level.append(summary)

        current_level = next_level

    return current_level[0]

final_summary = hierarchical_summarize([doc.page_content for doc in docs])
```

## Refine Strategy

Iteratively refine summary with each chunk.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

# Initial summary prompt
initial_prompt = ChatPromptTemplate.from_template("""
Create an initial summary of the following text:

{chunk}

Summary:
""")

# Refine prompt
refine_prompt = ChatPromptTemplate.from_template("""
Here is an existing summary:
{current_summary}

Here is additional content:
{chunk}

Refine the summary to incorporate this new information.
If the new content isn't relevant, return the original summary.

Refined Summary:
""")

initial_chain = initial_prompt | llm | StrOutputParser()
refine_chain = refine_prompt | llm | StrOutputParser()

def refine_summarize(docs):
    # Initial summary from first chunk
    current_summary = initial_chain.invoke({
        "chunk": docs[0].page_content
    })

    # Refine with each subsequent chunk
    for doc in docs[1:]:
        current_summary = refine_chain.invoke({
            "current_summary": current_summary,
            "chunk": doc.page_content,
        })

    return current_summary

summary = refine_summarize(docs)
```

### Refine with Early Stopping

```python
def refine_with_stopping(docs, max_refinements=10):
    current_summary = initial_chain.invoke({"chunk": docs[0].page_content})

    for i, doc in enumerate(docs[1:max_refinements + 1]):
        new_summary = refine_chain.invoke({
            "current_summary": current_summary,
            "chunk": doc.page_content,
        })

        # Stop if summary didn't change significantly
        if similarity(current_summary, new_summary) > 0.95:
            break

        current_summary = new_summary

    return current_summary
```

## Conversation Summarization

Summarize chat history to maintain context.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

summarize_prompt = ChatPromptTemplate.from_template("""
Summarize the following conversation, preserving key information
and context that would be needed for future responses.

Conversation:
{conversation}

Summary:
""")

summarize_chain = summarize_prompt | ChatOpenAI(model="gpt-4o") | StrOutputParser()

def format_conversation(messages):
    formatted = []
    for msg in messages:
        role = msg.type.capitalize()
        formatted.append(f"{role}: {msg.content}")
    return "\n".join(formatted)

# Summarize when conversation gets too long
def maybe_summarize(messages, max_messages=20):
    if len(messages) > max_messages:
        # Summarize older messages
        old_messages = messages[:-10]
        recent_messages = messages[-10:]

        summary = summarize_chain.invoke({
            "conversation": format_conversation(old_messages)
        })

        # Return summary + recent messages
        return [
            SystemMessage(content=f"Previous conversation summary: {summary}"),
            *recent_messages
        ]

    return messages
```

### Rolling Summary

```python
class ConversationWithSummary:
    def __init__(self, max_messages=10):
        self.messages = []
        self.summary = ""
        self.max_messages = max_messages

    def add_message(self, message):
        self.messages.append(message)

        if len(self.messages) > self.max_messages:
            self._update_summary()

    def _update_summary(self):
        # Summarize oldest messages
        to_summarize = self.messages[:-5]
        self.messages = self.messages[-5:]

        new_summary = summarize_chain.invoke({
            "conversation": format_conversation(to_summarize),
            "existing_summary": self.summary,
        })

        self.summary = new_summary

    def get_context(self):
        context = []
        if self.summary:
            context.append(f"Summary: {self.summary}")
        context.extend([format_message(m) for m in self.messages])
        return "\n".join(context)
```

## Structured Summaries

Get summaries with specific structure.

```python
from pydantic import BaseModel, Field
from typing import List

class DocumentSummary(BaseModel):
    title: str = Field(description="Document title or topic")
    main_points: List[str] = Field(description="Key points (3-5)")
    conclusion: str = Field(description="Main conclusion")
    keywords: List[str] = Field(description="Important keywords")

llm = ChatOpenAI(model="gpt-4o")
structured_llm = llm.with_structured_output(DocumentSummary)

prompt = ChatPromptTemplate.from_template("""
Analyze and summarize this document:

{document}
""")

chain = prompt | structured_llm

summary = chain.invoke({"document": doc.page_content})
print(summary.main_points)  # ["Point 1", "Point 2", ...]
```

## Best Practices

1. **Match strategy to document size** - Stuff for small, map-reduce for large
2. **Preserve important metadata** - Source attribution matters
3. **Tune chunk sizes** - Too small loses context, too large misses details
4. **Test with real documents** - Different content types need different prompts
5. **Consider output format** - Structured output for programmatic use
6. **Use async for performance** - Parallel map operations
