"""The benchmark topics: `evals/topics.jsonl`, one JSON object per line."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ValidationError, field_validator

TOPICS_PATH = Path("evals/topics.jsonl")
EXPECTED_TOPICS = 12
MIN_POINTS, MAX_POINTS = 3, 5


class Topic(BaseModel):
    id: str
    topic: str
    domain: str
    must_cover: list[str]

    @field_validator("id", "topic", "domain")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()

    @field_validator("must_cover")
    @classmethod
    def _points(cls, value: list[str]) -> list[str]:
        if not MIN_POINTS <= len(value) <= MAX_POINTS:
            raise ValueError(f"needs {MIN_POINTS} to {MAX_POINTS} must_cover points")
        if any(not p.strip() for p in value):
            raise ValueError("a must_cover point is blank")
        return [p.strip() for p in value]


class TopicsError(ValueError):
    """The topics file is not valid; the message says which line."""


def parse_topics(text: str, expected: int | None = EXPECTED_TOPICS) -> list[Topic]:
    topics: list[Topic] = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            topics.append(Topic.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise TopicsError(f"topics line {number}: {exc}") from exc
    ids = [t.id for t in topics]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise TopicsError(f"duplicate topic ids: {', '.join(duplicates)}")
    if expected is not None and len(topics) != expected:
        raise TopicsError(f"expected {expected} topics, found {len(topics)}")
    return topics


def load_topics(path: Path = TOPICS_PATH, expected: int | None = EXPECTED_TOPICS) -> list[Topic]:
    return parse_topics(path.read_text(encoding="utf-8"), expected)
