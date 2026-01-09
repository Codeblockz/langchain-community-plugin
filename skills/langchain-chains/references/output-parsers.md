# Output Parsers

## Table of Contents
- [Overview](#overview)
- [StrOutputParser](#stroutputparser)
- [JsonOutputParser](#jsonoutputparser)
- [PydanticOutputParser](#pydanticoutputparser)
- [CommaSeparatedListOutputParser](#commaseparatedlistoutputparser)
- [Custom Parsers](#custom-parsers)
- [Error Handling](#error-handling)

## Overview

Output parsers transform LLM responses into structured data.

| Parser | Output Type | Use Case |
|--------|-------------|----------|
| `StrOutputParser` | `str` | Plain text responses |
| `JsonOutputParser` | `dict` | JSON data without schema |
| `PydanticOutputParser` | Pydantic model | Validated structured data |
| `CommaSeparatedListOutputParser` | `List[str]` | Lists of items |

## StrOutputParser

Extract text content from AIMessage.

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()

# Direct use
response = llm.invoke("Say hello")
text = parser.invoke(response)  # "Hello!"

# In chain
chain = llm | parser
text = chain.invoke("Say hello")  # "Hello!"
```

### With Prompts

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("Summarize: {text}")

chain = prompt | llm | StrOutputParser()
summary = chain.invoke({"text": "Long document..."})
```

## JsonOutputParser

Parse JSON from LLM output.

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

parser = JsonOutputParser()

prompt = ChatPromptTemplate.from_template("""
Extract the following information as JSON:
- name
- age
- occupation

Text: {text}

Return ONLY valid JSON.
""")

chain = prompt | llm | parser

result = chain.invoke({"text": "John is a 30 year old engineer."})
print(result)  # {"name": "John", "age": 30, "occupation": "engineer"}
```

### With Pydantic Schema

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    occupation: str

parser = JsonOutputParser(pydantic_object=Person)

# Get format instructions for prompt
format_instructions = parser.get_format_instructions()

prompt = ChatPromptTemplate.from_template("""
Extract person information.

{format_instructions}

Text: {text}
""")

chain = prompt | llm | parser
```

## PydanticOutputParser

Parse into validated Pydantic models.

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

class Recipe(BaseModel):
    name: str = Field(description="Recipe name")
    ingredients: List[str] = Field(description="List of ingredients")
    instructions: List[str] = Field(description="Step by step instructions")
    prep_time: int = Field(description="Prep time in minutes")

parser = PydanticOutputParser(pydantic_object=Recipe)

prompt = ChatPromptTemplate.from_template("""
Create a recipe for {dish}.

{format_instructions}
""")

chain = prompt.partial(
    format_instructions=parser.get_format_instructions()
) | llm | parser

recipe = chain.invoke({"dish": "chocolate cake"})
print(recipe.name)         # "Chocolate Cake"
print(recipe.ingredients)  # ["flour", "sugar", ...]
```

### Nested Models

```python
class Ingredient(BaseModel):
    name: str
    amount: str
    unit: str

class DetailedRecipe(BaseModel):
    name: str
    ingredients: List[Ingredient]
    instructions: List[str]

parser = PydanticOutputParser(pydantic_object=DetailedRecipe)
```

## CommaSeparatedListOutputParser

Parse comma-separated lists.

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()

prompt = ChatPromptTemplate.from_template("""
List 5 programming languages as a comma-separated list.
{format_instructions}
""")

chain = prompt.partial(
    format_instructions=parser.get_format_instructions()
) | llm | parser

languages = chain.invoke({})
print(languages)  # ["Python", "JavaScript", "Java", "C++", "Go"]
```

## Custom Parsers

### Function-Based Parser

```python
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import RunnableLambda

def parse_bullet_points(text: str) -> List[str]:
    """Parse bullet point list."""
    lines = text.strip().split('\n')
    points = []
    for line in lines:
        line = line.strip()
        if line.startswith(('- ', '* ', '• ')):
            points.append(line[2:])
        elif line.startswith(tuple(f"{i}." for i in range(1, 10))):
            points.append(line.split('.', 1)[1].strip())
    return points

# Use as lambda
parser = RunnableLambda(parse_bullet_points)
chain = prompt | llm | StrOutputParser() | parser
```

### Class-Based Parser

```python
from langchain_core.output_parsers import BaseOutputParser
from typing import List

class BulletPointParser(BaseOutputParser[List[str]]):
    """Parse bullet point lists."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split('\n')
        return [
            line.strip().lstrip('- *•').strip()
            for line in lines
            if line.strip().startswith(('- ', '* ', '• '))
        ]

    def get_format_instructions(self) -> str:
        return "Return your response as a bullet point list using '- ' prefix."

parser = BulletPointParser()
```

### Chained Parsers

```python
from langchain_core.runnables import RunnableLambda

def clean_json(text: str) -> str:
    """Extract JSON from markdown code blocks."""
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    return text.strip()

# Chain parsers
chain = (
    prompt
    | llm
    | StrOutputParser()
    | RunnableLambda(clean_json)
    | JsonOutputParser()
)
```

## Error Handling

### OutputFixingParser

Automatically fix parsing errors.

```python
from langchain.output_parsers import OutputFixingParser
from langchain_openai import ChatOpenAI

# Wrap parser with fixing capability
base_parser = PydanticOutputParser(pydantic_object=Person)
fixing_parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=ChatOpenAI(model="gpt-4o")
)

# Will attempt to fix malformed output
result = fixing_parser.invoke("Name: John, age: thirty")  # Fixes "thirty" → 30
```

### RetryWithErrorOutputParser

Retry with error feedback.

```python
from langchain.output_parsers import RetryWithErrorOutputParser

retry_parser = RetryWithErrorOutputParser.from_llm(
    parser=base_parser,
    llm=llm,
    max_retries=3,
)

# Includes error message in retry prompt
result = retry_parser.invoke(malformed_output)
```

### Manual Error Handling

```python
from langchain_core.exceptions import OutputParserException

def safe_parse(text, parser):
    try:
        return parser.invoke(text)
    except OutputParserException as e:
        print(f"Parsing failed: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# With fallback
def parse_with_fallback(text, parser, fallback):
    try:
        return parser.invoke(text)
    except:
        return fallback
```

### Validation in Chain

```python
from langchain_core.runnables import RunnableLambda

def validate_output(data: dict) -> dict:
    """Validate and clean extracted data."""
    if not data.get("name"):
        raise ValueError("Name is required")

    # Clean/normalize data
    data["name"] = data["name"].strip().title()

    if data.get("age"):
        data["age"] = int(data["age"])

    return data

chain = prompt | llm | JsonOutputParser() | RunnableLambda(validate_output)
```

## Best Practices

1. **Include format instructions** - Use `parser.get_format_instructions()`
2. **Handle parsing failures** - Use OutputFixingParser or manual error handling
3. **Validate outputs** - Don't trust LLM output blindly
4. **Use appropriate parser** - Match parser to your output format
5. **Chain parsers for complex outputs** - Clean → Parse → Validate
6. **Test with edge cases** - Empty responses, malformed output, unexpected formats
