"""PlatoAI web application entrypoint."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = os.environ.get(
    "PLATOAI_DATA_PATH",
    os.path.join("data", "processed", "music_survey_data_with_embeddings.csv"),
)
DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "PLATOAI_EMBED_MODEL", "text-embedding-3-large"
)
DEFAULT_DIALOGUE_MODEL = os.environ.get("PLATOAI_DIALOGUE_MODEL", "gpt-5")
TOP_N_RESULTS = int(os.environ.get("PLATOAI_TOP_N", "50"))
HISTORY_MESSAGE_LIMIT = int(os.environ.get("PLATOAI_HISTORY_LIMIT", "60"))

EMBEDDABLE_COLUMNS: Dict[str, Dict[str, str]] = {
    "Q3_artist_that_pulled_you_in": {
        "summary": "What was the first song or artist that really pulled you in?",
        "question": "What was the first song or artist that really pulled you in?",
    },
    "Q5_Music_formal_change_impact": {
        "summary": "What do you remember about the shift from your earlier music format to the new one — or how it changed the way you listened?",
        "question": "What do you remember about the shift from your earlier music format to the new one — or how it changed the way you listened?",
    },
    "Q16_Music_guilty_pleasure_text_OE": {
        "summary": "What’s your music guilty pleasure? Spill it 👀",
        "question": "What’s your music guilty pleasure? Spill it 👀",
    },
    "Q18_Life_theme_song": {
        "summary": "If your life had a theme song right now, what would it be — and why?",
        "question": "If your life had a theme song right now, what would it be — and why?",
    },
    "Q19_Lyric_that_stuck_with_you": {
        "summary": "What’s one song lyric that stuck with you or changed the way you see the world? Share it and who wrote it!",
        "question": "What’s one song lyric that stuck with you or changed the way you see the world? Share it and who wrote it!",
    },
}


@dataclass
class Source:
    id: int
    question_text: str
    full_question: str
    verbatim_text: str
    persona: str
    respondent_name: str

    def to_prompt_line(self) -> str:
        return (
            f"Source {self.id}: {self.persona} {self.question_text} "
            f'They said: "{self.verbatim_text}"'
        )


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found. Set it before starting the app.")

client = OpenAI(api_key=api_key)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


def _safe_parse_embedding(value: object) -> Optional[np.ndarray]:
    if isinstance(value, (list, tuple)):
        arr = np.asarray(value, dtype=np.float32)
        return arr if arr.size else None
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    string_value = str(value).strip()
    if not string_value or string_value == "[]":
        return None
    try:
        parsed = json.loads(string_value)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    arr = np.asarray(parsed, dtype=np.float32)
    return arr if arr.size else None


def _format_persona(row: pd.Series) -> str:
    age_value = row.get("Age")
    age_phrase: Optional[str] = None
    if pd.notna(age_value):
        try:
            age_phrase = f"{int(round(float(age_value)))}-year-old"
        except (TypeError, ValueError):
            age_phrase = str(age_value).strip()
    gender_value = str(row.get("Gender", "")).strip()
    gender_phrase = gender_value if gender_value else "listener"
    region_value = str(row.get("Region", "")).strip()
    region_phrase = region_value if region_value else "an unspecified region"
    parts = ["A"]
    if age_phrase:
        parts.append(age_phrase)
    parts.append(gender_phrase)
    parts.append(f"from {region_phrase}.")
    return " ".join(parts)


def _format_name(row: pd.Series) -> str:
    name_value = str(row.get("distribution_name", "")).strip()
    if name_value and not (name_value.isupper() and len(name_value) <= 4):
        return name_value
    participant_id = str(row.get("participant_id", "")).strip()
    if participant_id:
        return f"Respondent {participant_id[:8]}"
    return "Anonymous respondent"


def _prepare_embeddings(df: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
    prepared: Dict[str, Dict[str, np.ndarray]] = {}
    for column in EMBEDDABLE_COLUMNS:
        embed_column = f"{column}_embedding"
        vectors: List[Optional[np.ndarray]] = [
            _safe_parse_embedding(value) for value in df.get(embed_column, [])
        ]
        example = next((vec for vec in vectors if vec is not None), None)
        if example is None:
            prepared[column] = {
                "matrix": np.zeros((len(df), 0), dtype=np.float32),
                "mask": np.zeros(len(df), dtype=bool),
            }
            continue
        dimension = example.shape[0]
        matrix = np.zeros((len(df), dimension), dtype=np.float32)
        mask = np.zeros(len(df), dtype=bool)
        for idx, vector in enumerate(vectors):
            if vector is not None and vector.shape[0] == dimension:
                matrix[idx] = vector
                mask[idx] = True
        prepared[column] = {"matrix": matrix, "mask": mask}
    return prepared


def _build_source(row: pd.Series, column: str, source_id: int) -> Source:
    raw_text = str(row.get(column, ""))
    verbatim = " ".join(raw_text.split())
    persona = _format_persona(row)
    question_meta = EMBEDDABLE_COLUMNS[column]
    question_text = question_meta["summary"]
    full_question = question_meta["question"]
    respondent_name = _format_name(row)
    return Source(
        id=source_id,
        question_text=question_text,
        full_question=full_question,
        verbatim_text=verbatim,
        persona=persona,
        respondent_name=respondent_name,
    )


def get_embedding(text: str, model: str = DEFAULT_EMBEDDING_MODEL) -> np.ndarray:
    safe_text = text.strip()
    if not safe_text:
        raise ValueError("Cannot embed an empty question.")
    response = client.embeddings.create(input=[safe_text], model=model)
    return np.asarray(response.data[0].embedding, dtype=np.float32)


def find_most_relevant_responses(
    query_embedding: np.ndarray, top_n: int = TOP_N_RESULTS
) -> List[Source]:
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    scored_results: List[Dict[str, object]] = []
    for column, payload in EMBEDDING_VECTORS.items():
        matrix = payload["matrix"]
        mask = payload["mask"]
        if matrix.size == 0 or not mask.any():
            continue
        active_indices = np.where(mask)[0]
        active_matrix = matrix[mask]
        scores = cosine_similarity(query_embedding, active_matrix)[0]
        for offset, score in enumerate(scores):
            row_index = int(active_indices[offset])
            scored_results.append(
                {
                    "score": float(score),
                    "row_index": row_index,
                    "column": column,
                }
            )
    scored_results.sort(key=lambda item: item["score"], reverse=True)
    selected: List[Source] = []
    seen_keys = set()
    for result in scored_results:
        key = (result["row_index"], result["column"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        row = DATAFRAME.iloc[result["row_index"]]
        source = _build_source(row, result["column"], len(selected) + 1)
        selected.append(source)
        if len(selected) >= top_n:
            break
    return selected


def _sanitize_history(raw_history: object) -> List[Dict[str, str]]:
    if not isinstance(raw_history, list):
        return []
    clean: List[Dict[str, str]] = []
    for item in raw_history:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role not in {"user", "assistant"}:
            continue
        if not content:
            continue
        clean.append({"role": role, "content": content})
    if HISTORY_MESSAGE_LIMIT and len(clean) > HISTORY_MESSAGE_LIMIT:
        clean = clean[-HISTORY_MESSAGE_LIMIT:]
    return clean


def _history_to_context(history: List[Dict[str, str]]) -> str:
    if not history:
        return ""
    lines: List[str] = []
    for message in history:
        role = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{role}: {message['content']}")
    return "\n".join(lines)


def _build_prompt(
    sources: List[Source], user_question: str, conversation_context: str
) -> Dict[str, str]:
    if not sources:
        raise ValueError("No sources available to build the prompt.")
    context_lines = [source.to_prompt_line() for source in sources]
    context_for_prompt = "\n".join(context_lines)
    history_block = conversation_context.strip() or "None."
    system_prompt = """
You are channeling Plato's distinctive voice to craft a living dialogue between Socrates and Glaucon.

**Character Roles:**
- **Socrates:** A data-grounded philosopher who probes toward the Form of truth. You wield the evidence provided in the 'Survey Data Context' as if it were testimony in the agora, paraphrasing or quoting it carefully. Invite Glaucon to examine assumptions, employ analogies, define terms, and trace causes. Never inject knowledge beyond the supplied sources.
- **Glaucon:** A spirited interlocutor whose curiosity mirrors the reader’s doubts. You raise common intuitions, press for clarification, and allow Socrates to expose contradictions or unveil deeper harmonies.

**Dialectical Tone Guidance:**
- Keep the exchange contemplative, lucid, and purposeful—as in Plato's dialogues.
- Let Socrates guide with gentle irony, measured patience, and stepwise reasoning.
- Let Glaucon respond with wonder, occasional skepticism, and willingness to be led toward insight.

**CRITICAL RULE FOR CITATIONS:**
When Socrates references information from a specific source in the 'Survey Data Context', conclude that sentence with a citation marker in the exact format `[cite:ID]`, where `ID` is the number of the source. If multiple sources inform the sentence, include each in the marker (e.g., `[cite:1][cite:3]`). Glaucon NEVER uses citations.

**Output Formatting Requirements:**
- Output must be plain text, not Markdown.
- Each line must begin with either `Socrates:` or `Glaucon:` followed by a single space and their dialogue.
- Do not include blank lines between turns or any commentary before or after the dialogue.

**Other Rules:**
1.  Address the 'User's Question' directly while weaving philosophical insight from the provided evidence.
2.  Base every claim on the 'Survey Data Context'. Do not invent facts or speculate beyond the supplied material.
3.  Alternate between Socrates and Glaucon, beginning with Socrates.
4.  Keep every line on a new line with the required `Name: ` prefix.
""".strip()
    user_prompt = f"""
**Conversation History:**
{history_block}

**User's Question:** "{user_question}"

**Survey Data Context:**
{context_for_prompt}

Begin the dialogue now.
""".strip()
    return {"system": system_prompt, "user": user_prompt}


def generate_dialogue(
    user_question: str, sources: List[Source], conversation_context: str
) -> str:
    prompts = _build_prompt(sources, user_question, conversation_context)
    response = client.chat.completions.create(
        model=DEFAULT_DIALOGUE_MODEL,
        messages=[
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompts["user"]},
        ],
        # temperature=0.7,
    )
    return response.choices[0].message.content.strip()


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/generate_dialogue", methods=["POST"])
def handle_generate_dialogue():
    payload = request.get_json(force=True, silent=True) or {}
    user_question = str(payload.get("question", "")).strip()
    raw_history = payload.get("conversation_history", [])
    history = _sanitize_history(raw_history)
    conversation_context = _history_to_context(history)
    if not user_question:
        return jsonify({"error": "Please provide a question."}), 400
    try:
        query_text = user_question
        if conversation_context:
            combined = f"{conversation_context}\n{user_question}"
            query_text = combined[-8000:]
        query_embedding = get_embedding(query_text)
        sources = find_most_relevant_responses(query_embedding)
        if not sources:
            return jsonify({"error": "No relevant survey responses found."}), 404
        dialogue_text = generate_dialogue(user_question, sources, conversation_context)
    except Exception as exc:  # noqa: BLE001 - surface full exception
        return jsonify({"error": str(exc)}), 500
    sources_payload = {str(source.id): source.__dict__ for source in sources}
    return jsonify({"dialogue": dialogue_text, "sources": sources_payload})


def _generate_processed_data() -> None:
    script_path = os.path.join(
        os.path.dirname(__file__), "scripts", "embeddings_of_data.py"
    )
    if not os.path.exists(script_path):
        raise FileNotFoundError(
            "Unable to generate processed data because embeddings_of_data.py was not found."
        )
    try:
        subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError as exc:  # noqa: BLE001 - propagate context
        raise RuntimeError(
            "Failed to generate processed survey embeddings. Check the script output for details."
        ) from exc


def _load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        _generate_processed_data()
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(
                f"Processed data not found at {DATA_PATH} even after generation attempt"
            )
    return pd.read_csv(DATA_PATH)


DATAFRAME = _load_data()
EMBEDDING_VECTORS = _prepare_embeddings(DATAFRAME)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
