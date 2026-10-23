from enum import Enum

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel


class QuestionCategory(str, Enum):
    FACTUAL_DIRECT_LOOKUP = "Factual direct lookup"
    BROAD_QUESTION = "Broad question"


class QuestionClassification(BaseModel):
    category: QuestionCategory


CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Classify the user's question about the ABB ACS880 firmware manual into exactly one category.

Factual direct lookup: asks for a specific fact, value, definition, setting, parameter, or single procedure.
Broad question: asks for an overview, explanation, comparison, multiple related facts, or how several things work together.

Return only the category using the supplied structured output schema.""",
        ),
        ("human", "{question}"),
    ]
)


def classify_question(question: str, llm) -> QuestionCategory:
    if not question or not question.strip():
        raise ValueError("Please enter a question.")
    classifier = llm.with_structured_output(QuestionClassification)
    result = classifier.invoke(
        CLASSIFICATION_PROMPT.format_messages(question=question.strip())
    )
    return result.category
