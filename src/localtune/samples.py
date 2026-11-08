"""Sample answers: the same 20 questions put to many models, so people can compare them."""

# Before downloading a model, people want to know how it writes and how clever it is. So every
# model gets the same 20 questions (data/sample_questions.json), from level 1 (a birthday
# message) to level 20 (a counting puzzle), and its answers are saved. The site shows them side
# by side: the same question makes the differences the model's own. Some levels have one right
# answer (a number, or a word that must appear) and are marked right or wrong here; the others
# are there to read.
#
# The answers come from Ollama, which downloads the model, loads it onto the graphics card (and
# into system memory when it doesn't fit) and answers. Each model keeps its own recommended
# settings, with the same fixed seed for all, so a re-run gives the same answers. A model that
# thinks before answering is allowed to, as it would in use; only its answer is kept. Answers
# go into samples.jsonl, beside results.jsonl, one per line, so a stopped run carries on where
# it left off. Models downloaded only for this are deleted again after their 20 answers, to
# keep the disk free; models that were already there are left alone.

import json
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from localtune import __version__
from localtune.models import ModelSpec, load_model_table
from localtune.results import home_dir

SEED = 42
CONTEXT = 8192
MAX_WORDS = 1536  # tokens for an answer
MAX_THINKING = 4096  # tokens for a model that thinks first, thinking included
ANSWER_WAIT_S = 1800


class Question(BaseModel):
    level: int
    id: str
    kind: Literal["read", "number", "contains"]
    ask: str
    answer: float | list[str] | None = None


class SampleAnswer(BaseModel):
    model: str
    question: str
    level: int
    kind: str
    answer: str
    correct: bool | None = None
    thought: bool = False
    tokens: int | None = None
    seconds: float | None = None
    stopped_early: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    localtune_version: str = __version__


def load_questions() -> list[Question]:
    text = (files("localtune") / "data" / "sample_questions.json").read_text(encoding="utf-8")
    return [Question(**q) for q in json.loads(text)]


# A question with one right answer asks for it on a last line of its own, so it can be found
# among the explaining. Without that line, the last number in the answer counts.

ANSWER_LINE = "\n\nEnd with a last line that says only: Answer: <the number>"


def prompt_for(question: Question) -> str:
    return question.ask + (ANSWER_LINE if question.kind == "number" else "")


NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def final_number(text: str) -> float | None:
    lines = [line for line in text.splitlines() if "answer" in line.lower()]
    for where in (lines[-1:] if lines else [], [text]):
        found = NUMBER.findall(" ".join(where).replace("**", ""))
        if found:
            try:
                return float(found[-1].replace(",", ""))
            except ValueError:
                return None
    return None


def mark(question: Question, text: str) -> bool | None:
    if question.kind == "number":
        number = final_number(text)
        return number is not None and abs(number - float(question.answer)) < 1e-6
    if question.kind == "contains":
        lowered = text.lower()
        return any(word in lowered for word in question.answer)
    return None


THINKING = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)


def without_thinking(text: str) -> str:
    return THINKING.sub("", text).strip()


# Which models: the most downloaded ones whose default download fits in this PC's graphics
# memory and system memory together, leaving room for the system. Only one size per name is
# taken, the one Ollama downloads for that exact name.


def models_that_fit(memory_gb: float, how_many: int) -> list[ModelSpec]:
    fitting = [
        m
        for m in load_model_table()
        if m.files and m.files[0].ollama == m.name and m.files[0].size_bytes / 1e9 <= memory_gb
    ]
    fitting.sort(key=lambda m: (m.popularity_rank, m.params))
    return fitting[:how_many]


# Saved answers, and which ones are done.


def samples_file() -> Path:
    return home_dir() / "samples.jsonl"


def save_answer(answer: SampleAnswer, path: Path | None = None) -> None:
    path = path or samples_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(answer.model_dump_json() + "\n")


def load_answers(path: Path | None = None) -> list[SampleAnswer]:
    path = path or samples_file()
    if not path.is_file():
        return []
    found = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            found.append(SampleAnswer.model_validate_json(line))
        except ValueError:
            continue
    return found


def done_pairs(answers: list[SampleAnswer]) -> set[tuple[str, str]]:
    return {(a.model, a.question) for a in answers}


# Talking to Ollama.


class AskError(Exception):
    pass


def call(
    url: str, path: str, body: dict | None, timeout_s: float = 60, method: str = "POST"
) -> dict:
    request = urllib.request.Request(
        url + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            text = response.read()
    except urllib.error.HTTPError as error:
        try:
            message = json.loads(error.read())["error"]
        except (OSError, ValueError, KeyError, TypeError):
            message = f"error {error.code}"
        raise AskError(str(message)) from error
    except (urllib.error.URLError, OSError) as error:
        raise AskError(f"Ollama didn't answer ({error})") from error
    return json.loads(text) if text.strip() else {}


# Whether Ollama already has a model, asked of Ollama itself, since it is Ollama that deletes the
# models "localtune samples" downloaded; a model the person had before is never deleted. A name
# with no tag means its "latest" tag, the way Ollama lists it.


def ollama_has(url: str, model: str) -> bool:
    listed = call(url, "/api/tags", None, timeout_s=10, method="GET").get("models") or []
    wanted = model if ":" in model.rsplit("/", 1)[-1] else f"{model}:latest"
    return any(wanted in (entry.get("name"), entry.get("model")) for entry in listed)


def can_think(url: str, model: str) -> bool:
    try:
        shown = call(url, "/api/show", {"model": model})
    except AskError:
        return False
    return "thinking" in (shown.get("capabilities") or [])


def ask(url: str, model: str, question: Question, thinks: bool) -> SampleAnswer:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_for(question)}],
        "stream": False,
        "options": {
            "seed": SEED,
            "num_ctx": CONTEXT,
            "num_predict": MAX_THINKING if thinks else MAX_WORDS,
        },
    }
    if thinks:
        body["think"] = True
    try:
        reply = call(url, "/api/chat", body, timeout_s=ANSWER_WAIT_S)
    except AskError:
        # Some thinking models take a thinking level rather than yes; they think anyway
        if not thinks:
            raise
        del body["think"]
        reply = call(url, "/api/chat", body, timeout_s=ANSWER_WAIT_S)
    text = without_thinking((reply.get("message") or {}).get("content", ""))
    took = reply.get("total_duration")
    return SampleAnswer(
        model=model,
        question=question.id,
        level=question.level,
        kind=question.kind,
        answer=text,
        correct=mark(question, text),
        thought=thinks,
        tokens=reply.get("eval_count"),
        seconds=round(took / 1e9, 1) if took else None,
        stopped_early=reply.get("done_reason") == "length",
    )


def unload(url: str, model: str) -> None:
    try:
        call(url, "/api/generate", {"model": model, "keep_alive": 0})
    except AskError:
        pass


def delete(url: str, model: str) -> None:
    call(url, "/api/delete", {"model": model}, method="DELETE")


# The scorecard for one model: how many questions with one right answer it got, and the
# highest level it got right.


def score(answers: list[SampleAnswer]) -> tuple[int, int, int | None]:
    checked = [a for a in answers if a.correct is not None]
    right = [a for a in checked if a.correct]
    return len(right), len(checked), max((a.level for a in right), default=None)
