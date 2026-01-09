# Extraction Patterns

## Table of Contents
- [Structured Output Basics](#structured-output-basics)
- [Pydantic Models](#pydantic-models)
- [Entity Extraction](#entity-extraction)
- [Data Extraction](#data-extraction)
- [Schema Design](#schema-design)
- [Validation](#validation)

## Structured Output Basics

Force LLM to return data in a specific format.

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age in years")
    occupation: str = Field(description="Person's job or role")

llm = ChatOpenAI(model="gpt-4o")
structured_llm = llm.with_structured_output(Person)

result = structured_llm.invoke("John Smith is a 35 year old software engineer.")
print(result.name)        # "John Smith"
print(result.age)         # 35
print(result.occupation)  # "software engineer"
```

## Pydantic Models

### Basic Model

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class Article(BaseModel):
    """Extracted article information."""
    title: str = Field(description="Article title")
    author: str = Field(description="Author name")
    date: str = Field(description="Publication date")
    summary: str = Field(description="Brief summary")
    topics: List[str] = Field(description="Main topics covered")
```

### Nested Models

```python
class Address(BaseModel):
    street: str
    city: str
    country: str
    postal_code: Optional[str] = None

class Company(BaseModel):
    name: str
    industry: str
    headquarters: Address
    founded_year: int

class Person(BaseModel):
    name: str
    role: str
    company: Company
```

### With Validators

```python
from pydantic import BaseModel, Field, field_validator

class Product(BaseModel):
    name: str
    price: float = Field(gt=0, description="Price must be positive")
    category: str

    @field_validator('category')
    @classmethod
    def validate_category(cls, v):
        allowed = ['electronics', 'clothing', 'food', 'other']
        if v.lower() not in allowed:
            raise ValueError(f"Category must be one of {allowed}")
        return v.lower()
```

### Using TypedDict

```python
from typing_extensions import TypedDict, Annotated

class Movie(TypedDict):
    title: str
    year: Annotated[int, "Release year"]
    director: str
    rating: Annotated[float, "Rating out of 10"]

llm = ChatOpenAI(model="gpt-4o")
structured_llm = llm.with_structured_output(Movie)
```

## Entity Extraction

### Named Entity Recognition

```python
from pydantic import BaseModel, Field
from typing import List

class Entity(BaseModel):
    text: str = Field(description="The entity text")
    type: str = Field(description="Entity type: PERSON, ORG, LOCATION, DATE, etc.")
    context: str = Field(description="Surrounding context")

class Entities(BaseModel):
    entities: List[Entity] = Field(description="All extracted entities")

llm = ChatOpenAI(model="gpt-4o")
extractor = llm.with_structured_output(Entities)

text = "Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976."
result = extractor.invoke(f"Extract all entities from: {text}")

for entity in result.entities:
    print(f"{entity.text} ({entity.type})")
# Apple Inc. (ORG)
# Steve Jobs (PERSON)
# Cupertino, California (LOCATION)
# 1976 (DATE)
```

### Relationship Extraction

```python
class Relationship(BaseModel):
    subject: str = Field(description="First entity")
    predicate: str = Field(description="Relationship type")
    object: str = Field(description="Second entity")

class ExtractedRelationships(BaseModel):
    relationships: List[Relationship]

extractor = llm.with_structured_output(ExtractedRelationships)

text = "Elon Musk is the CEO of Tesla and SpaceX."
result = extractor.invoke(f"Extract relationships from: {text}")
# relationships: [
#   {subject: "Elon Musk", predicate: "CEO of", object: "Tesla"},
#   {subject: "Elon Musk", predicate: "CEO of", object: "SpaceX"}
# ]
```

## Data Extraction

### From Tables

```python
from typing import List

class TableRow(BaseModel):
    columns: List[str] = Field(description="Values for each column")

class Table(BaseModel):
    headers: List[str] = Field(description="Column headers")
    rows: List[TableRow] = Field(description="Table rows")

extractor = llm.with_structured_output(Table)

# Extract from markdown or text table
table_text = """
| Name | Age | City |
|------|-----|------|
| John | 30  | NYC  |
| Jane | 25  | LA   |
"""

result = extractor.invoke(f"Extract the table:\n{table_text}")
```

### From Forms

```python
class FormField(BaseModel):
    label: str
    value: str
    field_type: str = Field(description="text, number, date, checkbox, etc.")

class FormData(BaseModel):
    form_title: str
    fields: List[FormField]

extractor = llm.with_structured_output(FormData)

form_text = """
Application Form

Name: John Doe
Email: john@example.com
Date of Birth: 1990-01-15
Agree to terms: Yes
"""

result = extractor.invoke(f"Extract form data:\n{form_text}")
```

### From Invoices

```python
from typing import Optional
from decimal import Decimal

class LineItem(BaseModel):
    description: str
    quantity: int
    unit_price: float
    total: float

class Invoice(BaseModel):
    invoice_number: str
    date: str
    vendor: str
    customer: str
    line_items: List[LineItem]
    subtotal: float
    tax: Optional[float] = None
    total: float

extractor = llm.with_structured_output(Invoice)
```

## Schema Design

### Optional vs Required Fields

```python
from typing import Optional

class Article(BaseModel):
    # Required fields
    title: str
    content: str

    # Optional fields with defaults
    author: Optional[str] = None
    published_date: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
```

### Enums for Constrained Values

```python
from enum import Enum

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class Review(BaseModel):
    text: str
    sentiment: Sentiment
    confidence: float = Field(ge=0, le=1)
```

### Multiple Return Types

```python
from typing import Union

class SuccessResult(BaseModel):
    success: bool = True
    data: dict

class ErrorResult(BaseModel):
    success: bool = False
    error: str

class ExtractionResult(BaseModel):
    result: Union[SuccessResult, ErrorResult]
```

## Validation

### Include Raw Response

```python
# Get both parsed and raw response
structured_llm = llm.with_structured_output(Person, include_raw=True)

result = structured_llm.invoke("John is 30 years old.")
print(result["parsed"])       # Person object
print(result["raw"])          # Original AIMessage
print(result["parsing_error"]) # None if successful
```

### Handle Parsing Errors

```python
def safe_extract(text, schema, llm):
    structured_llm = llm.with_structured_output(schema, include_raw=True)

    result = structured_llm.invoke(text)

    if result["parsing_error"]:
        # Log error, retry, or return default
        print(f"Parsing error: {result['parsing_error']}")
        return None

    return result["parsed"]
```

### Retry on Failure

```python
from langchain_core.runnables import RunnableRetry

structured_llm = llm.with_structured_output(Person)
robust_extractor = structured_llm.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)
```

## Chain Integration

### Extraction in Chains

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Combine with prompt
prompt = ChatPromptTemplate.from_template("""
Extract information from this text:

{text}

Be thorough and include all relevant details.
""")

extraction_chain = prompt | llm.with_structured_output(Entity)

# Use with retrieval
chain = {
    "text": retriever | format_docs,
} | extraction_chain
```

### Multiple Extractions

```python
from langchain_core.runnables import RunnableParallel

parallel_extraction = RunnableParallel(
    entities=llm.with_structured_output(Entities),
    sentiment=llm.with_structured_output(Sentiment),
    summary=llm.with_structured_output(Summary),
)

result = parallel_extraction.invoke(text)
print(result["entities"])
print(result["sentiment"])
print(result["summary"])
```

## Best Practices

1. **Use descriptive Field descriptions** - Helps the model understand what to extract
2. **Start simple, add complexity** - Begin with basic schema, refine based on results
3. **Validate outputs** - Don't trust extracted data blindly
4. **Handle missing data** - Use Optional for fields that may not exist
5. **Test with edge cases** - Unusual input, missing information, ambiguous text
6. **Use include_raw for debugging** - See what the model actually returned
