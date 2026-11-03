import uuid
from typing import List

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Store a character-level pointer to a chunk within a source file."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Represent a dataset question that has not yet been answered."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represent a question with its ground-truth sources and answer."""

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Represent a collection of RAG benchmark questions."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Store the retrieval results for a single query."""

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Extend retrieval results for a single query with a generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Store batch retrieval results for a full question set."""

    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Store batch retrieval-and-answer results for a full question set."""

    search_results: list[MinimalAnswer]
    k: int
