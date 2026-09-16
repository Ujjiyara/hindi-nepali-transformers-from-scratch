"""Tests for the synthetic reasoning dataset.

A synthetic dataset is only worth finetuning on if its labels are right, and
"the generator built it that way" is not a check. So these tests re-derive the
answers from the *text* of each prompt -- reading the numbers and the
comparative sentences back out of it -- and compare them with the stored label.
They also confirm the leakage guarantees the report relies on.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from common.reasoning.generate import build_dataset

ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ("Hindi", "Nepali")
_TO_ASCII = str.maketrans("०१२३४५६७८९", "0123456789")


@pytest.fixture(scope="module", params=LANGUAGES)
def dataset(request):
    """A small dataset built from each language's real lexicon."""
    spec = yaml.safe_load(
        (ROOT / request.param / "configs" / "reasoning.yaml").read_text(encoding="utf-8")
    )
    spec["sizes"] = {"train": 800, "validation": 200, "test": 200, "test_long": 100}
    splits, stats = build_dataset(spec)
    return spec, splits, stats


def _mentions(text: str, names: list[str]) -> list[str]:
    """Names in the order they occur, longest alternative first.

    Longest-first matters: a Nepali name can be a substring of another
    (रिता inside सरिता), and a naive search would find the shorter one. Matches
    must also start a word -- "राम" occurs inside "किलोग्राम" -- but may be
    followed by a suffix, because Nepali postpositions attach ("रामको").
    """
    alternatives = "|".join(
        re.escape(n) for n in sorted(set(names), key=len, reverse=True)
    )
    pattern = rf"(?<![^\s,])(?:{alternatives})"
    return [m.group(0) for m in re.finditer(pattern, text)]


def _sentences(prompt: str) -> tuple[list[str], str]:
    """Split a prompt body into its fact sentences and the final question."""
    body = prompt.split(":", 1)[1].rsplit(" ", 1)[0]  # drop the frame
    parts = [p.strip() for p in body.split("।") if p.strip()]
    return parts[:-1], parts[-1]


def _adjectives(spec: dict, attribute: str, direction: str) -> set[str]:
    return set(spec["attributes"][attribute][direction].values())


def _direction(spec: dict, attribute: str, sentence: str) -> str:
    """Which comparative adjective a sentence uses."""
    words = set(re.split(r"[\s,?]+", sentence))
    greater = bool(words & _adjectives(spec, attribute, "greater"))
    smaller = bool(words & _adjectives(spec, attribute, "smaller"))
    assert greater != smaller, f"ambiguous direction in {sentence!r}"
    return "greater" if greater else "smaller"


def test_every_split_has_the_requested_size(dataset) -> None:
    spec, splits, _ = dataset
    for split, examples in splits.items():
        assert len(examples) == spec["sizes"][split]
        assert len({e["prompt"] for e in examples}) == len(examples)


def test_no_leakage_between_splits(dataset) -> None:
    """Disjoint names and disjoint prompts, as the report claims."""
    _, splits, stats = dataset
    leakage = stats["leakage"]
    assert leakage["name_overlap_train_test"] == 0
    assert leakage["name_overlap_train_validation"] == 0
    assert leakage["train_examples_mentioning_heldout_names"] == 0
    assert not any(leakage["prompt_overlap_with_train"].values())

    train_names = {n for e in splits["train"] for n in e["entities"]}
    for split in ("validation", "test", "test_long"):
        held_out = {n for e in splits[split] for n in e["entities"]}
        assert not train_names & held_out


def test_answers_are_among_the_choices(dataset) -> None:
    _, splits, _ = dataset
    for examples in splits.values():
        for e in examples:
            assert e["answer"] in e["choices"]
            assert len(set(e["choices"])) == len(e["choices"])


def test_numeric_labels_follow_from_the_stated_values(dataset) -> None:
    """Read the numbers back out of the text and solve the question."""
    spec, splits, _ = dataset
    for e in splits["train"]:
        if e["task"] not in ("numeric_pair", "numeric_triple"):
            continue
        facts, _ = _sentences(e["prompt"])
        values = {}
        for fact in facts:
            (name,) = _mentions(fact, e["entities"])
            (number,) = re.findall(r"[0-9०-९]+", fact)
            values[name] = int(number.translate(_TO_ASCII))

        if e["question"] == "equal":
            a, b = e["entities"]
            expected = spec["answer_yes"] if values[a] == values[b] else spec["answer_no"]
        elif len(set(values.values())) == 1:
            expected = spec["answer_equal"]
        else:
            pick = max if e["question"] == "greater" else min
            expected = pick(values, key=values.get)
        assert e["answer"] == expected, e["prompt"]


def test_chain_labels_follow_from_the_stated_comparisons(dataset) -> None:
    """Rebuild the ordering from the sentences alone, then answer from it."""
    spec, splits, _ = dataset
    for split in ("train", "test_long"):
        for e in splits[split]:
            if not e["task"].startswith("chain"):
                continue
            facts, question = _sentences(e["prompt"])
            if e["task"] == "chain_relation":
                # Relation questions end in "?" inside the last sentence.
                facts, question = facts + [question.split("?")[0]], None
                question = facts.pop()

            greater_than: set[tuple[str, str]] = set()
            for fact in facts:
                a, b = _mentions(fact, e["entities"])
                if _direction(spec, e["attribute"], fact) == "greater":
                    greater_than.add((a, b))
                else:
                    greater_than.add((b, a))

            # Transitive closure of the stated comparisons.
            changed = True
            while changed:
                changed = False
                for a, b in list(greater_than):
                    for c, d in list(greater_than):
                        if b == c and (a, d) not in greater_than:
                            greater_than.add((a, d))
                            changed = True

            names = set(e["entities"])
            beats = {n: sum((n, m) in greater_than for m in names) for n in names}
            if e["task"] == "chain_superlative":
                target = len(names) - 1 if e["question"] == "greater" else 0
                (expected,) = [n for n, wins in beats.items() if wins == target]
                assert e["answer"] == expected, e["prompt"]
            else:
                a, b = _mentions(question, e["entities"])
                holds = (a, b) in greater_than
                if _direction(spec, e["attribute"], question) == "smaller":
                    holds = (b, a) in greater_than
                expected = spec["answer_yes"] if holds else spec["answer_no"]
                assert e["answer"] == expected, e["prompt"]


def test_test_long_chains_are_longer_than_any_in_training(dataset) -> None:
    _, splits, _ = dataset
    longest_train = max(len(e["chain"]) for e in splits["train"] if "chain" in e)
    assert all(len(e["chain"]) > longest_train for e in splits["test_long"])


def test_yes_no_questions_are_balanced(dataset) -> None:
    spec, splits, _ = dataset
    relation = [e for e in splits["train"] if e["task"] == "chain_relation"]
    share = sum(e["answer"] == spec["answer_yes"] for e in relation) / len(relation)
    assert 0.4 < share < 0.6


def test_the_two_languages_are_drawn_independently() -> None:
    """Hindi and Nepali must not be one dataset translated line by line."""
    hops = {}
    for lang in LANGUAGES:
        spec = yaml.safe_load(
            (ROOT / lang / "configs" / "reasoning.yaml").read_text(encoding="utf-8")
        )
        spec["sizes"] = {"train": 200, "validation": 50, "test": 50, "test_long": 20}
        splits, _ = build_dataset(spec)
        hops[lang] = [(e["task"], e["attribute"], e["hops"]) for e in splits["train"]]
    assert hops["Hindi"] != hops["Nepali"]
