# LCEL Fundamentals

## Table of Contents
- [Pipe Operator](#pipe-operator)
- [Runnable Types](#runnable-types)
- [Composition Patterns](#composition-patterns)
- [Streaming](#streaming)
- [Error Handling](#error-handling)
- [Async Execution](#async-execution)

## Pipe Operator

The `|` operator connects components into chains.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Each component is a Runnable
prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
model = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()

# Pipe connects them: prompt → model → parser
chain = prompt | model | parser

# Invoke the chain
result = chain.invoke({"topic": "machine learning"})
```

### How It Works

```
Input: {"topic": "machine learning"}
    ↓
prompt.invoke() → "Tell me about machine learning"
    ↓
model.invoke() → AIMessage(content="Machine learning is...")
    ↓
parser.invoke() → "Machine learning is..."
    ↓
Output: "Machine learning is..."
```

## Runnable Types

### RunnablePassthrough

Passes input unchanged or adds to it.

```python
from langchain_core.runnables import RunnablePassthrough

# Pass input unchanged
chain = RunnablePassthrough() | prompt | model

# Assign additional values
chain = RunnablePassthrough.assign(
    processed=lambda x: x["text"].upper()
)
# Input: {"text": "hello"} → Output: {"text": "hello", "processed": "HELLO"}
```

### RunnableParallel

Execute multiple chains simultaneously.

```python
from langchain_core.runnables import RunnableParallel

# Define parallel branches
parallel = RunnableParallel(
    summary=prompt1 | model | parser,
    translation=prompt2 | model | parser,
)

# Both run at the same time
result = parallel.invoke({"text": "Document content"})
# {"summary": "...", "translation": "..."}
```

### RunnableLambda

Wrap any Python function.

```python
from langchain_core.runnables import RunnableLambda

def clean_text(text: str) -> str:
    return text.strip().lower()

# Sync function
cleaner = RunnableLambda(clean_text)

# Async function
async def async_fetch(url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

fetcher = RunnableLambda(async_fetch)
```

### RunnableBranch

Conditional routing based on input.

```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    # (condition, chain) pairs
    (lambda x: x["type"] == "question", question_chain),
    (lambda x: x["type"] == "summary", summary_chain),
    (lambda x: len(x["text"]) > 5000, long_text_chain),
    # Default (no condition)
    default_chain,
)

# Routes to appropriate chain
result = branch.invoke({"type": "question", "text": "What is AI?"})
```

### RunnableSequence

Explicit sequential composition.

```python
from langchain_core.runnables import RunnableSequence

# Equivalent to prompt | model | parser
chain = RunnableSequence(first=prompt, middle=[model], last=parser)
```

## Composition Patterns

### Dictionary Input Mapping

```python
# Map input keys to different values
chain = {
    "context": retriever,              # Call retriever with input
    "question": RunnablePassthrough(), # Pass input unchanged
    "history": lambda x: x.get("history", []),  # Extract with default
} | prompt | model
```

### Chaining with .pipe()

```python
# Alternative to | operator
chain = prompt.pipe(model).pipe(parser)

# Mix with additional processing
chain = prompt.pipe(model).pipe(
    RunnableLambda(lambda x: x.content.upper())
)
```

### Nested Chains

```python
# Inner chain
inner = prompt1 | model | parser

# Outer chain uses inner
outer = {
    "preprocessed": inner,
    "original": RunnablePassthrough(),
} | prompt2 | model
```

### Binding Arguments

```python
# Bind constant arguments to model
model_with_temp = model.bind(temperature=0.0)
model_with_stop = model.bind(stop=["\n\n"])

# Use in chain
chain = prompt | model_with_temp | parser
```

## Streaming

Stream output as it's generated.

### Basic Streaming

```python
# Stream chunks
for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="", flush=True)
```

### Async Streaming

```python
async for chunk in chain.astream({"topic": "AI"}):
    print(chunk, end="", flush=True)
```

### Stream Events

```python
# Get detailed events during execution
async for event in chain.astream_events({"topic": "AI"}, version="v2"):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="")
```

### Streaming with Intermediate Steps

```python
# Stream logs from intermediate steps
async for chunk in chain.astream_log({"topic": "AI"}):
    print(chunk)
```

## Error Handling

### Fallbacks

```python
# Try main chain, fall back on error
chain_with_fallback = main_chain.with_fallbacks([
    backup_chain1,
    backup_chain2,
])

# Fallback with exception handling
chain = main_chain.with_fallbacks(
    [backup_chain],
    exceptions_to_handle=(ValueError, TimeoutError),
)
```

### Retry

```python
from langchain_core.runnables import RunnableRetry

# Retry on failure
chain_with_retry = chain.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)
```

### Config with Timeout

```python
from langchain_core.runnables import RunnableConfig

# Set timeout
config = RunnableConfig(max_concurrency=5, timeout=30)
result = chain.invoke({"topic": "AI"}, config=config)
```

## Async Execution

### Async Invoke

```python
# Single async call
result = await chain.ainvoke({"topic": "AI"})
```

### Batch Processing

```python
# Sync batch
results = chain.batch([
    {"topic": "AI"},
    {"topic": "ML"},
    {"topic": "DL"},
])

# Async batch
results = await chain.abatch([
    {"topic": "AI"},
    {"topic": "ML"},
])

# With concurrency limit
results = chain.batch(
    inputs,
    config={"max_concurrency": 5}
)
```

### Parallel with Gather

```python
import asyncio

# Run multiple chains in parallel
results = await asyncio.gather(
    chain1.ainvoke({"topic": "AI"}),
    chain2.ainvoke({"topic": "ML"}),
    chain3.ainvoke({"topic": "DL"}),
)
```

## Configuration

### Configurable Fields

```python
from langchain_core.runnables import ConfigurableField

# Make model configurable
model = ChatOpenAI(model="gpt-4o").configurable_fields(
    model_name=ConfigurableField(
        id="model_name",
        name="Model Name",
        description="The model to use",
    )
)

chain = prompt | model | parser

# Override at runtime
result = chain.invoke(
    {"topic": "AI"},
    config={"configurable": {"model_name": "gpt-4o-mini"}}
)
```

### Configurable Alternatives

```python
from langchain_core.runnables import ConfigurableField
from langchain_anthropic import ChatAnthropic

# Define alternatives
model = ChatOpenAI(model="gpt-4o").configurable_alternatives(
    ConfigurableField(id="model"),
    default_key="openai",
    anthropic=ChatAnthropic(model="claude-3-5-sonnet-20241022"),
)

# Switch at runtime
result = chain.invoke(
    {"topic": "AI"},
    config={"configurable": {"model": "anthropic"}}
)
```

## Best Practices

1. **Use type hints** - Makes chains easier to understand
2. **Keep chains flat** - Deeply nested chains are hard to debug
3. **Name your chains** - Use `.with_config(run_name="my_chain")`
4. **Add fallbacks for production** - Graceful degradation
5. **Use streaming for long outputs** - Better user experience
6. **Batch when possible** - More efficient than sequential calls
