"""Generate a synthetic comparative-reasoning dataset for one language.

The brief asks for reasoning data built programmatically, so that every label
is known by construction rather than trusted from an annotator. Each example is
assembled from a per-language lexicon (``<Language>/configs/reasoning.yaml``):
entity names with a grammatical gender, measurable attributes, and sentence
templates written from that language's grammar. The generator itself knows no
Hindi or Nepali -- it only fills slots -- which is what keeps the two datasets
separate while letting them test the same skills.

**Four task families**, in equal proportion:

``numeric_pair``
    Two entities with stated values. Asks which is greater or smaller, or
    whether the two are equal. A fifth of greater/smaller pairs are given equal
    values, so "both equal" is a real answer and cannot be ruled out by
    default; equality questions are balanced between yes and no.
``numeric_triple``
    Three entities with distinct stated values; asks for the largest or the
    smallest. Needs a comparison across all three, not just two.
``chain_superlative``
    No numbers at all -- only pairwise comparisons that link into a chain
    (A > B, B > C, ...). Asks for the extreme of the chain. The comparisons
    are stated in a shuffled order and each one randomly uses the "greater" or
    the "smaller" adjective (``A > B`` may be written "B is shorter than A"),
    so the answer cannot be read off the first or last sentence.
``chain_relation``
    The same chains, but asks a yes/no question about two entities that are
    usually *not* compared directly: most questions need two or more hops of
    transitivity.

**Avoiding train-test leakage.** Entity names are partitioned into disjoint
train / validation / test pools *before* any example is generated, per gender,
so a test example never mentions a name seen in training. A model can only
score on the test split by applying the comparison, not by recalling facts
about particular names. Exact prompt overlap between splits is checked as well
and is an error. A fourth split, ``test_long``, uses the test names with
chains one entity longer than anything in training (five entities, four
hops), to measure whether the chaining generalises beyond the lengths seen.

**Answer balance.** Mention order is shuffled independently of the value
order, and yes/no questions are drawn to be balanced, so neither "always the
first name" nor "always yes" beats chance. The statistics file records the
majority-class baseline for each family so the accuracy numbers can be read
against it.

Run from the repository root::

    python -m common.reasoning.generate \
        --config Hindi/configs/reasoning.yaml \
        --out Hindi/data/reasoning \
        --tokenizer Hindi/tokenizer/hi_unigram_16000.model
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

log = logging.getLogger("reasoning")

TASKS = ("numeric_pair", "numeric_triple", "chain_superlative", "chain_relation")
SPLITS = ("train", "validation", "test", "test_long")

# Chain lengths per split. test_long is deliberately outside the training range.
CHAIN_LENGTHS = {
    "train": (3, 4),
    "validation": (3, 4),
    "test": (3, 4),
    "test_long": (5,),
}

_DEVANAGARI_DIGITS = str.maketrans("0123456789", "०१२३४५६७८९")


@dataclass(frozen=True)
class Entity:
    """A person or object that can be compared.

    Attributes:
        name: The surface form, in the target language's script.
        gender: ``"m"`` or ``"f"``; selects adjective and copula forms.
    """

    name: str
    gender: str


class Language:
    """One language's lexicon and templates, and the slot-filling around them."""

    def __init__(self, spec: dict) -> None:
        self.spec = spec
        self.lang = spec["lang"]
        self.attributes: dict[str, dict] = spec["attributes"]
        self.copula: dict[str, str] = spec.get("copula", {"m": "", "f": ""})

    def number(self, value: int) -> str:
        """Write a value in the digit system this language's corpus uses."""
        text = str(value)
        if self.spec.get("digits") == "devanagari":
            return text.translate(_DEVANAGARI_DIGITS)
        return text

    def adjective(self, attribute: str, direction: str, gender: str) -> str:
        """The comparative adjective agreeing with a subject of ``gender``."""
        return self.attributes[attribute][direction][gender]

    def fill(self, template: str, **slots) -> str:
        """Fill a template, supplying the copula and feminine adjective slots."""
        subject_gender = slots.pop("gender", "m")
        slots.setdefault("cop", self.copula.get(subject_gender, ""))
        return template.format(**slots)

    def fact(self, attribute: str, entity: Entity, value: int, variant: int) -> str:
        """State one entity's value, choosing the gender-appropriate template."""
        spec = self.attributes[attribute]
        templates = spec.get("fact_f") if entity.gender == "f" else None
        templates = templates or spec["fact"]
        return self.fill(templates[variant % len(templates)], name=entity.name,
                         value=self.number(value), gender=entity.gender)

    def compare(self, attribute: str, a: Entity, b: Entity, direction: str) -> str:
        """``a`` is ``direction`` (greater/smaller) than ``b``."""
        spec = self.attributes[attribute]
        return self.fill(spec["compare"], a=a.name, b=b.name,
                         adj=self.adjective(attribute, direction, a.gender),
                         gender=a.gender)

    def ask(self, key: str, attribute: str, direction: str,
            a: Entity | None = None, b: Entity | None = None) -> str:
        """A question from one of the attribute's ``ask_*`` templates.

        Questions whose subject is unknown ("who is tallest?") take the
        masculine forms, the unmarked choice in both languages; questions
        about a named subject agree with that subject.
        """
        spec = self.attributes[attribute]
        gender = a.gender if key == "ask_relation" and a is not None else "m"
        return self.fill(
            spec[key],
            a=a.name if a else "", b=b.name if b else "",
            adj=self.adjective(attribute, direction, gender),
            adj_f=self.adjective(attribute, direction, "f"),
            gender=gender,
        )

    def prompt(self, body: str) -> str:
        """Wrap a question body in the fixed prompt frame the model sees."""
        return self.spec["prompt"].format(body=body)


def split_entities(spec: dict, seed: int) -> dict[str, dict[str, list[Entity]]]:
    """Partition every entity pool into disjoint train/validation/test pools.

    Done per pool and per gender, so each split keeps both genders and both
    entity kinds. Validation and test get at least five entities from every
    pool so that the five-entity chains of ``test_long`` can always be built.

    Returns:
        ``{split: {"people": [...], "objects": [...]}}``.
    """
    # Seeded with the language too, so the two datasets are drawn independently
    # rather than being translations of one another example for example.
    rng = random.Random(f"{seed}-{spec['lang']}-names")
    fractions = spec["name_split"]
    pools: dict[str, dict[str, list[Entity]]] = {
        s: {"people": [], "objects": []} for s in ("train", "validation", "test")
    }

    for kind in ("people", "objects"):
        by_gender = spec.get(kind, {})
        for gender in sorted(by_gender):
            names = list(dict.fromkeys(by_gender[gender]))  # dedupe, keep order
            rng.shuffle(names)
            n = len(names)
            n_val = max(1, round(n * fractions["validation"]))
            n_test = max(1, round(n * fractions["test"]))
            parts = {
                "validation": names[:n_val],
                "test": names[n_val:n_val + n_test],
                "train": names[n_val + n_test:],
            }
            for split, chunk in parts.items():
                pools[split][kind].extend(Entity(name, gender) for name in chunk)

    for split, kinds in pools.items():
        for kind, entities in kinds.items():
            if split != "train" and len(entities) < 5:
                raise ValueError(
                    f"{split} has only {len(entities)} {kind}; need at least 5 "
                    "for five-entity chains -- add names to the lexicon"
                )
    return pools


class ExampleBuilder:
    """Builds individual examples from one split's entity pools."""

    def __init__(self, language: Language, pools: dict[str, list[Entity]],
                 rng: random.Random) -> None:
        self.language = language
        self.pools = pools
        self.rng = rng

    # --- helpers -------------------------------------------------------------

    def _attribute(self) -> str:
        return self.rng.choice(sorted(self.language.attributes))

    def _entities(self, attribute: str, count: int) -> list[Entity]:
        kind = self.language.attributes[attribute]["entities"]
        return self.rng.sample(self.pools[kind], count)

    def _values(self, attribute: str, count: int, distinct: bool = True) -> list[int]:
        low, high = self.language.attributes[attribute]["range"]
        step = self.language.attributes[attribute].get("step", 1)
        candidates = range(low, high + 1, step)
        if distinct:
            return self.rng.sample(candidates, count)
        return [self.rng.choice(candidates) for _ in range(count)]

    def _record(self, task, attribute, question, body, answer, choices,
                entities, template, hops, chain=None) -> dict:
        record = {
            "task": task,
            "attribute": attribute,
            "question": question,
            "hops": hops,
            "prompt": self.language.prompt(body),
            "answer": answer,
            "choices": choices,
            "entities": [e.name for e in entities],
            "template": template,
        }
        if chain is not None:
            record["chain"] = [e.name for e in chain]
        return record

    # --- task families -------------------------------------------------------

    def numeric_pair(self) -> dict:
        """Two stated values; greater, smaller, or equal?"""
        lang, rng = self.language, self.rng
        attribute = self._attribute()
        a, b = self._entities(attribute, 2)
        kind = rng.choice(("greater", "smaller", "equal"))
        # Equality questions are balanced yes/no; for greater/smaller questions
        # a fifth of pairs are tied, so "both equal" is a live answer there too.
        equal = rng.random() < (0.5 if kind == "equal" else 0.2)
        va, vb = self._values(attribute, 2)
        if equal:
            vb = va

        variant = rng.randrange(2)
        facts = [lang.fact(attribute, a, va, variant),
                 lang.fact(attribute, b, vb, variant)]

        if kind == "equal":
            question = lang.ask("ask_equal", attribute, "greater", a, b)
            answer = lang.spec["answer_yes"] if va == vb else lang.spec["answer_no"]
            choices = [lang.spec["answer_yes"], lang.spec["answer_no"]]
        else:
            question = lang.ask("ask_pair", attribute, kind, a, b)
            if va == vb:
                answer = lang.spec["answer_equal"]
            else:
                pick_a = (va > vb) == (kind == "greater")
                answer = a.name if pick_a else b.name
            choices = [a.name, b.name, lang.spec["answer_equal"]]

        return self._record("numeric_pair", attribute, kind,
                            " ".join(facts) + " " + question, answer, choices,
                            [a, b], f"fact{variant}|{kind}", 1)

    def numeric_triple(self) -> dict:
        """Three distinct stated values; which is the largest or smallest?"""
        lang, rng = self.language, self.rng
        attribute = self._attribute()
        entities = self._entities(attribute, 3)
        values = self._values(attribute, 3)
        kind = rng.choice(("greater", "smaller"))
        key = rng.choice(("ask_most", "ask_most_alt"))

        facts = [lang.fact(attribute, e, v, rng.randrange(2))
                 for e, v in zip(entities, values)]
        pick = max if kind == "greater" else min
        answer = entities[values.index(pick(values))].name

        return self._record("numeric_triple", attribute, kind,
                            " ".join(facts) + " " + lang.ask(key, attribute, kind),
                            answer, [e.name for e in entities], entities,
                            f"{key}|{kind}", 2)

    def _chain(self, attribute: str,
               length: int) -> tuple[list[Entity], list[str], list[Entity]]:
        """Entities in descending order, shuffled facts, and mention order.

        ``chain[0]`` is the greatest. Each adjacent pair becomes one sentence,
        written either as "chain[i] is greater than chain[i+1]" or as
        "chain[i+1] is smaller than chain[i]", at random.
        """
        chain = self._entities(attribute, length)
        facts = []
        for upper, lower in zip(chain, chain[1:]):
            if self.rng.random() < 0.5:
                facts.append((self.language.compare(attribute, upper, lower, "greater"),
                              (upper, lower)))
            else:
                facts.append((self.language.compare(attribute, lower, upper, "smaller"),
                              (lower, upper)))
        self.rng.shuffle(facts)
        mentioned = list(dict.fromkeys(e for _, pair in facts for e in pair))
        return chain, [text for text, _ in facts], mentioned

    def chain_superlative(self, lengths: tuple[int, ...]) -> dict:
        """Pairwise comparisons only; find the extreme of the chain."""
        lang, rng = self.language, self.rng
        attribute = self._attribute()
        length = rng.choice(lengths)
        chain, facts, mentioned = self._chain(attribute, length)
        kind = rng.choice(("greater", "smaller"))
        key = rng.choice(("ask_most", "ask_most_alt"))
        answer = chain[0].name if kind == "greater" else chain[-1].name

        choices = [e.name for e in chain]
        rng.shuffle(choices)
        return self._record("chain_superlative", attribute, kind,
                            " ".join(facts) + " " + lang.ask(key, attribute, kind),
                            answer, choices, mentioned, f"{key}|{kind}", length - 1,
                            chain=chain)

    def chain_relation(self, lengths: tuple[int, ...]) -> dict:
        """Pairwise comparisons; a yes/no question needing transitivity.

        Four times in five the two entities asked about are not adjacent in
        the chain, so no single stated sentence answers the question.
        """
        lang, rng = self.language, self.rng
        attribute = self._attribute()
        length = rng.choice(lengths)
        chain, facts, mentioned = self._chain(attribute, length)

        if rng.random() < 0.8:
            i, j = sorted(rng.sample(range(length), 2))
            while j - i < 2:
                i, j = sorted(rng.sample(range(length), 2))
        else:
            i = rng.randrange(length - 1)
            j = i + 1

        # chain[i] is greater than chain[j]. Ask about the pair in either
        # order, with either adjective, and the truth follows.
        want_yes = rng.random() < 0.5
        direction = rng.choice(("greater", "smaller"))
        greater_first = (direction == "greater") == want_yes
        a, b = (chain[i], chain[j]) if greater_first else (chain[j], chain[i])
        answer = lang.spec["answer_yes"] if want_yes else lang.spec["answer_no"]

        question = lang.ask("ask_relation", attribute, direction, a, b)
        return self._record("chain_relation", attribute, direction,
                            " ".join(facts) + " " + question, answer,
                            [lang.spec["answer_yes"], lang.spec["answer_no"]], mentioned,
                            f"relation|{direction}", j - i, chain=chain)


def generate_split(language: Language, pools: dict[str, list[Entity]],
                   split: str, size: int, seed: int,
                   exclude: set[str] | None = None) -> list[dict]:
    """Generate one split with the four task families in equal proportion.

    Args:
        exclude: Prompts already used by other splits; an example whose
            prompt is in here is redrawn rather than kept.

    Returns:
        ``size`` examples with unique prompts, in generation order.
    """
    # A different stream per split: seeding every split identically would
    # correlate their structure even with disjoint names.
    rng = random.Random(f"{seed}-{language.lang}-{split}")
    builder = ExampleBuilder(language, pools, rng)
    lengths = CHAIN_LENGTHS[split]
    exclude = exclude or set()

    makers = {
        "numeric_pair": builder.numeric_pair,
        "numeric_triple": builder.numeric_triple,
        "chain_superlative": lambda: builder.chain_superlative(lengths),
        "chain_relation": lambda: builder.chain_relation(lengths),
    }
    # test_long only makes sense for the chain tasks.
    tasks = ("chain_superlative", "chain_relation") if split == "test_long" else TASKS

    examples, seen = [], set()
    attempts = 0
    while len(examples) < size:
        attempts += 1
        if attempts > size * 50:
            raise RuntimeError(f"{split}: could not find {size} unique prompts")
        task = tasks[len(examples) % len(tasks)]
        example = makers[task]()
        if example["prompt"] in seen or example["prompt"] in exclude:
            continue
        seen.add(example["prompt"])
        example["id"] = f"{language.lang}-{split}-{len(examples):06d}"
        examples.append(example)

    rng.shuffle(examples)
    return examples


def dataset_statistics(splits: dict[str, list[dict]], pools: dict,
                       spec: dict, sp=None) -> dict:
    """Sizes, composition, template variety, leakage checks and baselines."""
    stats: dict = {"lang": spec["lang"], "seed": spec["seed"], "splits": {}}

    names = {
        split: {e.name for kind in pools[split].values() for e in kind}
        for split in ("train", "validation", "test")
    }
    stats["entities"] = {
        split: {kind: len(entities) for kind, entities in pools[split].items()}
        for split in names
    }

    for split, examples in splits.items():
        by_task = Counter(e["task"] for e in examples)
        majority = {}
        for task in by_task:
            answers = Counter(e["answer"] for e in examples if e["task"] == task)
            # For name answers the majority class is meaningless (names vary),
            # so the baseline is uniform guessing over the offered choices.
            task_examples = [e for e in examples if e["task"] == task]
            chance = sum(1 / len(e["choices"]) for e in task_examples) / len(task_examples)
            majority[task] = {
                "chance": chance,
                "most_common_answer_share": answers.most_common(1)[0][1] / len(task_examples),
            }

        entry = {
            "examples": len(examples),
            "by_task": dict(sorted(by_task.items())),
            "by_attribute": dict(sorted(Counter(e["attribute"] for e in examples).items())),
            "by_hops": dict(sorted(Counter(e["hops"] for e in examples).items())),
            "templates": len({(e["task"], e["attribute"], e["template"]) for e in examples}),
            "baselines": majority,
            "mean_prompt_chars": sum(len(e["prompt"]) for e in examples) / len(examples),
            # How often the answer is the first-mentioned entity; ~chance if
            # mention order carries no signal.
            "answer_is_first_entity": sum(
                e["answer"] == e["entities"][0] for e in examples
                if e["answer"] in e["entities"]
            ) / max(1, sum(e["answer"] in e["entities"] for e in examples)),
        }
        if sp is not None:
            lengths = [len(sp.encode(e["prompt"])) + len(sp.encode(e["answer"])) + 1
                       for e in examples]
            entry["tokens_mean"] = sum(lengths) / len(lengths)
            entry["tokens_max"] = max(lengths)
        stats["splits"][split] = entry

    train_prompts = {e["prompt"] for e in splits["train"]}
    stats["leakage"] = {
        "name_overlap_train_validation": len(names["train"] & names["validation"]),
        "name_overlap_train_test": len(names["train"] & names["test"]),
        "name_overlap_validation_test": len(names["validation"] & names["test"]),
        "prompt_overlap_with_train": {
            split: sum(e["prompt"] in train_prompts for e in examples)
            for split, examples in splits.items() if split != "train"
        },
        "train_examples_mentioning_heldout_names": sum(
            any(n in names["validation"] | names["test"] for n in e["entities"])
            for e in splits["train"]
        ),
    }
    return stats


def build_dataset(spec: dict, sp=None) -> tuple[dict[str, list[dict]], dict]:
    """Generate every split for one language and verify it does not leak."""
    language = Language(spec)
    seed = spec["seed"]
    pools = split_entities(spec, seed)

    splits: dict[str, list[dict]] = {}
    used: set[str] = set()
    for split in SPLITS:
        pool = pools["test" if split == "test_long" else split]
        splits[split] = generate_split(language, pool, split, spec["sizes"][split],
                                       seed, exclude=used)
        used |= {e["prompt"] for e in splits[split]}

    stats = dataset_statistics(splits, pools, spec, sp)
    leakage = stats["leakage"]
    if (leakage["name_overlap_train_test"] or leakage["name_overlap_train_validation"]
            or leakage["train_examples_mentioning_heldout_names"]
            or any(leakage["prompt_overlap_with_train"].values())):
        raise AssertionError(f"train/test leakage detected: {leakage}")
    return splits, stats


def write_jsonl(path: Path, records: list[dict]) -> None:
    """One JSON object per line, UTF-8, Devanagari unescaped."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    """Read a split written by :func:`write_jsonl`."""
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Generate synthetic reasoning data.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path,
                        help="directory for the split .jsonl files")
    parser.add_argument("--tokenizer", type=Path, default=None,
                        help="this language's tokenizer, to report token lengths")
    parser.add_argument("--stats", type=Path, default=None,
                        help="also write the statistics here (e.g. under runs/)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    spec = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    sp = None
    if args.tokenizer is not None:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))

    splits, stats = build_dataset(spec, sp)
    for split, examples in splits.items():
        write_jsonl(args.out / f"{split}.jsonl", examples)
        log.info("%s: %d examples", split, len(examples))

    text = json.dumps(stats, indent=2, ensure_ascii=False) + "\n"
    (args.out / "_stats.json").write_text(text, encoding="utf-8")
    if args.stats is not None:
        args.stats.parent.mkdir(parents=True, exist_ok=True)
        args.stats.write_text(text, encoding="utf-8")
    log.info("leakage checks: %s", stats["leakage"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
