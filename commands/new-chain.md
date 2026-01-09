---
name: new-chain
description: Scaffold a new LangChain LCEL chain for common patterns
allowed-tools:
  - Read
  - Write
  - AskUserQuestion
argument-hint: "[filename]"
---

# New Chain Command

Create a new LCEL chain file for common patterns.

## Workflow

1. **Ask the user** which chain type they want:
   - `summarization` - Summarize documents or text
   - `extraction` - Extract structured data
   - `classification` - Classify text into categories
   - `translation` - Translate between languages

2. **Get filename** from argument or ask user (default: `chain.py`)

3. **Generate the chain file** using the appropriate template below

4. **Inform user** about customization options

## Templates

### Summarization Template

```python
"""
Summarization Chain

Install: pip install langchain langchain-openai
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


def create_summarization_chain(
    model: str = "gpt-4o",
    style: str = "concise",
    max_points: int = 5,
):
    """
    Create a summarization chain.

    Args:
        model: The model to use
        style: Summary style (concise, detailed, bullet_points)
        max_points: Maximum bullet points for bullet_points style
    """
    style_instructions = {
        "concise": "Provide a brief, 2-3 sentence summary.",
        "detailed": "Provide a comprehensive summary covering all main points.",
        "bullet_points": f"Provide a summary as {max_points} bullet points.",
    }

    prompt = ChatPromptTemplate.from_template("""
Summarize the following text.

{style_instruction}

Text:
{text}

Summary:
""")

    llm = ChatOpenAI(model=model, temperature=0)

    chain = (
        prompt.partial(style_instruction=style_instructions.get(style, style_instructions["concise"]))
        | llm
        | StrOutputParser()
    )

    return chain


def main():
    # Create chain
    chain = create_summarization_chain(style="bullet_points", max_points=5)

    # Example usage
    text = """
    Artificial intelligence (AI) is transforming industries across the globe.
    From healthcare to finance, AI applications are improving efficiency and
    enabling new capabilities. Machine learning, a subset of AI, allows systems
    to learn from data without explicit programming. Deep learning, using neural
    networks, has achieved remarkable results in image and speech recognition.
    However, challenges remain around bias, interpretability, and ethical use.
    """

    summary = chain.invoke({"text": text})
    print(summary)


if __name__ == "__main__":
    main()
```

### Extraction Template

```python
"""
Data Extraction Chain

Install: pip install langchain langchain-openai pydantic
"""
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# Define your extraction schema
class Person(BaseModel):
    """Information about a person."""
    name: str = Field(description="Person's full name")
    role: Optional[str] = Field(None, description="Person's role or job title")
    organization: Optional[str] = Field(None, description="Organization they belong to")


class ExtractedData(BaseModel):
    """Extracted entities from text."""
    people: List[Person] = Field(default_factory=list, description="People mentioned")
    key_facts: List[str] = Field(default_factory=list, description="Key facts or statements")
    dates: List[str] = Field(default_factory=list, description="Dates mentioned")


def create_extraction_chain(
    schema: type[BaseModel] = ExtractedData,
    model: str = "gpt-4o",
):
    """
    Create an extraction chain.

    Args:
        schema: Pydantic model defining what to extract
        model: The model to use
    """
    prompt = ChatPromptTemplate.from_template("""
Extract information from the following text.
Be thorough and include all relevant details.

Text:
{text}
""")

    llm = ChatOpenAI(model=model, temperature=0)
    structured_llm = llm.with_structured_output(schema)

    chain = prompt | structured_llm

    return chain


def main():
    # Create chain
    chain = create_extraction_chain()

    # Example usage
    text = """
    On January 15, 2024, CEO Sarah Johnson announced that TechCorp
    would be acquiring DataInc for $2 billion. The deal is expected
    to close in Q2 2024. CTO Michael Chen will lead the integration
    effort. The acquisition will add 500 employees to TechCorp's
    workforce.
    """

    result = chain.invoke({"text": text})

    print("People found:")
    for person in result.people:
        print(f"  - {person.name}: {person.role} at {person.organization}")

    print("\nKey facts:")
    for fact in result.key_facts:
        print(f"  - {fact}")

    print("\nDates:")
    for date in result.dates:
        print(f"  - {date}")


if __name__ == "__main__":
    main()
```

### Classification Template

```python
"""
Text Classification Chain

Install: pip install langchain langchain-openai pydantic
"""
from enum import Enum
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# Define your categories
class Category(str, Enum):
    TECHNOLOGY = "technology"
    BUSINESS = "business"
    SCIENCE = "science"
    HEALTH = "health"
    ENTERTAINMENT = "entertainment"
    SPORTS = "sports"
    OTHER = "other"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class Classification(BaseModel):
    """Text classification result."""
    category: Category = Field(description="Primary category")
    secondary_categories: List[Category] = Field(
        default_factory=list,
        description="Additional relevant categories"
    )
    sentiment: Sentiment = Field(description="Overall sentiment")
    confidence: float = Field(
        ge=0, le=1,
        description="Confidence score between 0 and 1"
    )
    reasoning: str = Field(description="Brief explanation of classification")


def create_classification_chain(
    schema: type[BaseModel] = Classification,
    model: str = "gpt-4o",
):
    """
    Create a classification chain.

    Args:
        schema: Pydantic model defining classification output
        model: The model to use
    """
    prompt = ChatPromptTemplate.from_template("""
Classify the following text.

Categories available: {categories}
Sentiments: positive, negative, neutral

Text:
{text}
""")

    llm = ChatOpenAI(model=model, temperature=0)
    structured_llm = llm.with_structured_output(schema)

    # Get category names for prompt
    categories = ", ".join([c.value for c in Category])

    chain = prompt.partial(categories=categories) | structured_llm

    return chain


def main():
    # Create chain
    chain = create_classification_chain()

    # Example usage
    texts = [
        "The new iPhone 15 features an improved camera and faster processor.",
        "Stock market crashes amid recession fears, investors panic.",
        "Scientists discover high-temperature superconductor in meteor sample.",
    ]

    for text in texts:
        result = chain.invoke({"text": text})
        print(f"\nText: {text[:50]}...")
        print(f"  Category: {result.category.value}")
        print(f"  Sentiment: {result.sentiment.value}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Reasoning: {result.reasoning}")


if __name__ == "__main__":
    main()
```

### Translation Template

```python
"""
Translation Chain

Install: pip install langchain langchain-openai
"""
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


class TranslationResult(BaseModel):
    """Translation with metadata."""
    translated_text: str = Field(description="The translated text")
    source_language: str = Field(description="Detected source language")
    target_language: str = Field(description="Target language")
    notes: Optional[str] = Field(None, description="Translation notes or caveats")


def create_translation_chain(
    target_language: str = "English",
    model: str = "gpt-4o",
    structured: bool = False,
):
    """
    Create a translation chain.

    Args:
        target_language: Language to translate to
        model: The model to use
        structured: Return structured output with metadata
    """
    if structured:
        prompt = ChatPromptTemplate.from_template("""
Translate the following text to {target_language}.
Detect the source language and note any translation challenges.

Text:
{text}
""")
        llm = ChatOpenAI(model=model, temperature=0)
        structured_llm = llm.with_structured_output(TranslationResult)

        chain = prompt.partial(target_language=target_language) | structured_llm
    else:
        prompt = ChatPromptTemplate.from_template("""
Translate the following text to {target_language}.
Return only the translation, nothing else.

Text:
{text}

Translation:
""")
        llm = ChatOpenAI(model=model, temperature=0)

        chain = (
            prompt.partial(target_language=target_language)
            | llm
            | StrOutputParser()
        )

    return chain


def main():
    # Simple translation
    simple_chain = create_translation_chain(target_language="Spanish")
    result = simple_chain.invoke({"text": "Hello, how are you today?"})
    print(f"Simple: {result}")

    # Structured translation with metadata
    structured_chain = create_translation_chain(
        target_language="French",
        structured=True
    )
    result = structured_chain.invoke({"text": "The quick brown fox jumps over the lazy dog."})
    print(f"\nStructured:")
    print(f"  Translation: {result.translated_text}")
    print(f"  Source: {result.source_language}")
    print(f"  Notes: {result.notes}")


if __name__ == "__main__":
    main()
```

## After Generation

Tell the user:
1. Install dependencies listed in the docstring
2. Customize the schema/categories for their use case
3. Adjust the prompt for better results
4. Run with: `python <filename>`
