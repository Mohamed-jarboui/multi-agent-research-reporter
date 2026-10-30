"""Plan parsing, brief rendering and the deterministic merge."""

import json
import random
import re

from fakes import sections_json
from reporter.agent.multi import (
    generic_plan,
    merge_results,
    normalize_url,
    parse_plan,
    render_digest,
)
from reporter.agent.schemas import Finding, SectionBrief, WorkerResult
from reporter.agent.worker import render_brief
from reporter.tools import Page


def test_parse_plan_valid_sections_get_ordered_ids():
    briefs = parse_plan(sections_json(4), "T")
    assert [b.id for b in briefs] == ["s1", "s2", "s3", "s4"]
    assert briefs[0].title == "Part 1" and briefs[0].scope == ["point 1"]


def test_parse_plan_accepts_fenced_json():
    text = "Here you go:\n```json\n" + sections_json(3) + "\n```"
    assert len(parse_plan(text, "T")) == 3


def test_parse_plan_cuts_to_six():
    assert len(parse_plan(sections_json(9), "T")) == 6


def test_parse_plan_drops_invalid_items_and_duplicate_titles():
    good = {"title": "A", "goal": "g", "scope": ["x"]}
    items = [
        good,
        {"title": "a", "goal": "g2", "scope": ["y"]},  # duplicate title (casefold)
        {"title": "B", "goal": "g", "scope": []},  # no scope
        {"title": "", "goal": "g", "scope": ["x"]},  # blank title
        "not an object",
        {"title": "C", "goal": "g", "scope": "single point"},  # scope given as a string
    ]
    briefs = parse_plan(json.dumps({"sections": items}), "T")
    assert [b.title for b in briefs] == ["A", "C"]
    assert [b.id for b in briefs] == ["s1", "s2"]
    assert briefs[1].scope == ["single point"]


def test_parse_plan_without_any_valid_section_gives_three_generic_briefs():
    briefs = parse_plan("sorry, no JSON", "Solar power")
    assert [b.id for b in briefs] == ["s1", "s2", "s3"]
    assert briefs == generic_plan("Solar power")
    assert all("Solar power" in b.goal for b in briefs)


def test_brief_rendering_is_self_contained():
    brief = SectionBrief(
        id="s1",
        title="Costs",
        goal="How much does it cost?",
        scope=["unit costs", "trend"],
        exclude=["policy"],
        expected_output="3 numbers",
    )
    text = render_brief(brief, "Solar power", "2026-10-30")
    for part in [
        "Solar power",
        "2026-10-30",
        "Costs",
        "How much",
        "unit costs",
        "policy",
        "3 numbers",
    ]:
        assert part in text


def result(section: str, index: int, urls: list[str], findings: list[Finding], **kw):
    pages = [Page(url=u, title=u, text="t") for u in urls]
    return WorkerResult(
        section_id=section, index=index, status="complete", findings=findings, pages=pages, **kw
    )


def three_sections() -> list[SectionBrief]:
    return parse_plan(sections_json(3), "T")


def three_results() -> list[WorkerResult]:
    shared = "https://shared.example/a"
    return [
        result("s1", 0, ["https://one.example/", shared], [Finding(claim="c1", source_ids=[1, 2])]),
        result(
            "s2",
            1,
            ["https://SHARED.example/a/#frag", "https://two.example/"],
            [Finding(claim="c2", source_ids=[1, 2])],
            open_questions=["Why?", "Who?"],
        ),
        result(
            "s3",
            2,
            ["https://three.example/"],
            [Finding(claim="c3", source_ids=[1, 5])],  # 5 is not a page: dropped
            open_questions=["why?"],
        ),
    ]


def test_normalize_url():
    assert normalize_url("HTTPS://Example.COM/a/#x") == normalize_url("https://example.com/a")


def test_merge_gives_a_shared_url_one_global_id_and_remaps_findings():
    merged = merge_results(three_sections(), three_results())
    assert [s.url for s in merged.sources] == [
        "https://one.example/",
        "https://shared.example/a",
        "https://two.example/",
        "https://three.example/",
    ]
    ids = [[f.source_ids for f in s.findings] for s in merged.sections]
    assert ids == [[[1, 2]], [[2, 3]], [[4]]]
    assert merged.dropped_citations == 1
    assert merged.open_questions == [("s2", "Why?"), ("s2", "Who?")]  # "why?" deduplicated


def test_merge_numbering_does_not_depend_on_completion_order():
    results = three_results()
    expected = merge_results(three_sections(), results)
    for seed in range(5):
        shuffled = results[:]
        random.Random(seed).shuffle(shuffled)
        merged = merge_results(three_sections(), shuffled)
        assert merged.sources == expected.sources
        assert [s.findings for s in merged.sections] == [s.findings for s in expected.sections]


def test_merge_marks_a_section_without_result_as_incomplete():
    merged = merge_results(three_sections(), three_results()[:2])
    assert merged.incomplete == ["s3"]
    assert merged.sections[2].status == "incomplete"
    assert "INCOMPLETE" in render_digest("T", merged)


def test_digest_cites_only_single_numbers_that_are_listed_in_sources():
    merged = merge_results(three_sections(), three_results())
    digest = render_digest("T", merged)
    body, _, listing = digest.partition("## Sources")
    assert not re.search(r"\[\d+\s*,", body)
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", body)}
    listed = {int(n) for n in re.findall(r"^(\d+)\. ", listing, flags=re.M)}
    assert cited and cited <= listed
