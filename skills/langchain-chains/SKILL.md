---
name: langchain-chains
description: Build data processing pipelines with LangChain Expression Language (LCEL). Use when creating summarization, extraction, or multi-step chains without agents. Covers pipe operator composition, parallel execution, structured output, and output parsing patterns.
---

# LangChain Chains Builder

## Quick Decision: Chains vs Agents

| Use Chains when... | Use Agents when... |
|-------------------|-------------------|
| Fixed, predictable workflow | Dynamic decision-making needed |
| Single LLM call or fixed sequence | Multiple iterations, tool selection |
| Processing/transforming data | Interactive task completion |
| Summarization, extraction | Complex multi-step reasoning |
| Low latency critical | Flexibility more important |

## LCEL Quick Start

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Define components
prompt = ChatPromptTemplate.from_template(
    "Summarize this text in 3 bullet points:\n\n{text}"
)
llm = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()

# Compose with pipe operator
chain = prompt | llm | parser

# Invoke
result = chain.invoke({"text": "Long document here..."})
```

## Core LCEL Patterns

### RunnablePassthrough (Pass Input Through)

```python
from langchain_core.runnables import RunnablePassthrough

# Pass original input alongside processed value
chain = {
    "context": retriever,
    "question": RunnablePassthrough(),  # Passes input unchanged
} | prompt | llm
```

### RunnableParallel (Execute in Parallel)

```python
from langchain_core.runnables import RunnableParallel

# Run multiple chains simultaneously
parallel = RunnableParallel(
    summary=summarize_chain,
    keywords=extract_keywords_chain,
    sentiment=sentiment_chain,
)

# All run in parallel, results combined
result = parallel.invoke({"text": "Document content..."})
# {"summary": "...", "keywords": [...], "sentiment": "positive"}
```

### RunnableLambda (Custom Functions)

```python
from langchain_core.runnables import RunnableLambda

def process_text(text: str) -> str:
    return text.strip().lower()

# Wrap function as runnable
chain = RunnableLambda(process_text) | prompt | llm
```

### Branching (Conditional Routing)

```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: len(x["text"]) > 10000, long_doc_chain),
    (lambda x: "code" in x["text"], code_analysis_chain),
    default_chain,  # Fallback
)
```

## Structured Output

Get typed responses using Pydantic models.

```python
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class MovieReview(BaseModel):
    """Structured movie review."""
    title: str = Field(description="Movie title")
    rating: float = Field(description="Rating out of 10")
    summary: str = Field(description="Brief summary")
    pros: list[str] = Field(description="Positive points")
    cons: list[str] = Field(description="Negative points")

llm = ChatOpenAI(model="gpt-4o")
structured_llm = llm.with_structured_output(MovieReview)

review = structured_llm.invoke("Review the movie Inception")
print(review.rating)  # 9.2
print(review.pros)    # ["Innovative concept", "Great visuals", ...]
```

## Summarization Patterns

### Stuff (Small Documents)

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("""
Summarize the following documents:

{documents}

Summary:
""")

chain = prompt | ChatOpenAI(model="gpt-4o") | StrOutputParser()

# Works for content that fits in context window
result = chain.invoke({"documents": combined_text})
```

### Map-Reduce (Large Documents)

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel

# Map: Summarize each chunk
map_prompt = ChatPromptTemplate.from_template(
    "Summarize this section:\n\n{chunk}"
)
map_chain = map_prompt | llm | StrOutputParser()

# Reduce: Combine summaries
reduce_prompt = ChatPromptTemplate.from_template(
    "Combine these summaries into one:\n\n{summaries}"
)
reduce_chain = reduce_prompt | llm | StrOutputParser()

# Execute
chunk_summaries = [map_chain.invoke({"chunk": c}) for c in chunks]
final_summary = reduce_chain.invoke({"summaries": "\n\n".join(chunk_summaries)})
```

### Refine (Iterative)

```python
refine_prompt = ChatPromptTemplate.from_template("""
Current summary: {current_summary}

New information: {new_chunk}

Update the summary to incorporate this new information:
""")

current_summary = ""
for chunk in chunks:
    current_summary = (refine_prompt | llm | StrOutputParser()).invoke({
        "current_summary": current_summary,
        "new_chunk": chunk,
    })
```

## Critical Rules

1. **Chains are deterministic** - Same input → same execution path
2. **Use streaming for long outputs** - `chain.stream()` for better UX
3. **Handle errors** - Use `.with_fallbacks()` for resilience
4. **Keep chains simple** - Complex logic → use agents or LangGraph
5. **Type your inputs** - Use Pydantic models for validation

## Common Gotchas

### Missing input keys
```python
# WRONG - prompt expects "text" but receives "content"
prompt = ChatPromptTemplate.from_template("Summarize: {text}")
chain.invoke({"content": "..."})  # KeyError!

# CORRECT - match keys
chain.invoke({"text": "..."})
```

### Not awaiting async
```python
# WRONG - forgetting await
result = chain.ainvoke({"text": "..."})  # Returns coroutine, not result!

# CORRECT
result = await chain.ainvoke({"text": "..."})
```

## Reference Documentation

- **[lcel-fundamentals.md](references/lcel-fundamentals.md)** - Pipe operator, Runnables, composition
- **[summarization.md](references/summarization.md)** - Stuff, map-reduce, refine strategies
- **[extraction.md](references/extraction.md)** - Structured output, Pydantic models
- **[output-parsers.md](references/output-parsers.md)** - String, JSON, Pydantic parsers
