"""Retrieval tests.

Real embeddings against a real pgvector index — the thing being tested is
whether the right passage comes back for a plausibly-phrased question, and no
mock can answer that. The model loads once for the session.
"""

from __future__ import annotations

import pytest

from app.services import knowledge

POLICIES = {
    "Returns policy": (
        "Customers can return any item within 30 days of delivery for a full "
        "refund, as long as it is unused and in its original packaging."
    ),
    "Warranty": (
        "All desks and chairs carry a five year structural warranty. Lamps and "
        "accessories carry a two year warranty."
    ),
    "Delivery": (
        "Standard delivery is free on orders over 500 dollars and takes three "
        "to five working days. Next day delivery costs 25 dollars."
    ),
}


@pytest.fixture
async def indexed() -> None:
    for title, body in POLICIES.items():
        await knowledge.index_document(title=title, body=body)


async def test_finds_the_right_passage_for_a_paraphrased_question(
    indexed: None,
) -> None:
    """Callers never use the words in the document."""
    hits = await knowledge.search("can I send something back if I don't like it?")
    assert hits
    assert hits[0].title == "Returns policy"


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("how long is the guarantee on a desk?", "Warranty"),
        ("how much does next day shipping cost?", "Delivery"),
        ("what if I change my mind about an item?", "Returns policy"),
    ],
)
async def test_routes_questions_to_the_right_document(
    indexed: None, question: str, expected: str
) -> None:
    hits = await knowledge.search(question)
    assert hits and hits[0].title == expected


async def test_unrelated_question_returns_nothing(indexed: None) -> None:
    """The distance threshold's whole job.

    Without it the nearest neighbour always comes back, however unrelated, and
    the agent reads out a policy that has nothing to do with the question.
    """
    assert await knowledge.search("what is the capital of France?") == []


async def test_relevant_and_irrelevant_distances_stay_separated(
    indexed: None,
) -> None:
    """Pins the band MAX_DISTANCE was calibrated against.

    If a corpus change collapses this gap, the threshold needs re-tuning and
    this test should fail rather than the agent quietly answering nonsense.
    """
    relevant = await knowledge.search("can I get a refund?")
    assert relevant and relevant[0].distance < 0.45

    raw = await knowledge.search("the weather in Tashkent tomorrow", top_k=1)
    assert raw == [], "unrelated question should fall outside MAX_DISTANCE"


async def test_search_is_empty_when_nothing_is_indexed() -> None:
    """A fresh install must not error before the corpus is seeded."""
    assert await knowledge.search("returns policy") == []


async def test_reindexing_a_title_replaces_it(indexed: None) -> None:
    await knowledge.index_document(
        title="Returns policy", body="Returns are accepted within 60 days."
    )
    hits = await knowledge.search("how long do I have to return something?")
    assert hits[0].title == "Returns policy"
    assert "60 days" in hits[0].body


async def test_top_k_bounds_what_is_read_aloud(indexed: None) -> None:
    """Passages are spoken, so the count has to stay small."""
    assert len(await knowledge.search("delivery and returns", top_k=2)) <= 2
