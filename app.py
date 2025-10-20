"""PlatoAI web application entrypoint."""

from __future__ import annotations

import json
import os
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
DEFAULT_DIALOGUE_MODEL = os.environ.get("PLATOAI_DIALOGUE_MODEL", "gpt-5-mini")
TOP_N_RESULTS = int(os.environ.get("PLATOAI_TOP_N", "5"))

EMBEDDABLE_COLUMNS: Dict[str, str] = {
    "Q3_artist_that_pulled_you_in": "When asked about the first artist that captured their imagination...",
    "Q5_Music_formal_change_impact": "When reflecting on how a change in music formats impacted them...",
    "Q16_Music_guilty_pleasure_text_OE": "When revealing a musical guilty pleasure...",
    "Q18_Life_theme_song": "When sharing the theme song of their life right now...",
    "Q19_Lyric_that_stuck_with_you": "When discussing a lyric that stayed with them...",
}


@dataclass
class Source:
    id: int
    question_text: str
    verbatim_text: str
    persona: str

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
    question_text = EMBEDDABLE_COLUMNS[column]
    return Source(
        id=source_id,
        question_text=question_text,
        verbatim_text=verbatim,
        persona=persona,
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


def _build_prompt(sources: List[Source], user_question: str) -> Dict[str, str]:
    if not sources:
        raise ValueError("No sources available to build the prompt.")
    context_lines = [source.to_prompt_line() for source in sources]
    context_for_prompt = "\n".join(context_lines)
    system_prompt = """
You are a master of Socratic dialogue. You will generate a conversation between two characters: Socrates and Glaucon.

**Character Roles:**
- **Socrates:** You are a data-driven philosopher. You do not have personal opinions. Your role is to present evidence and insights found exclusively in the provided 'Survey Data Context'. You must guide the conversation by paraphrasing or quoting from the context.
- **Glaucon:** You are Socrates' curious student. You ask questions, express common assumptions, and react to the data Socrates presents.

**CRITICAL RULE FOR CITATIONS:**
When you, as Socrates, use information from a specific source in the 'Survey Data Context', you MUST end that sentence with a citation marker in the exact format `[cite:ID]`, where `ID` is the number of the source you are referencing. For example, if you are using information from 'Source 1', you must end your sentence with `[cite:1]`. Use multiple citations if a sentence draws from multiple sources, like `[cite:1]`. Glaucon NEVER uses citations.

**Other Rules:**
1.  The dialogue must directly address the 'User's Question'.
2.  Base the entire conversation on the 'Survey Data Context'. Do not invent facts.
3.  The dialogue should be philosophical, insightful, and easy to read.
4.  Alternate between Socrates and Glaucon. Start with Socrates.
5.  Format the output clearly, with each character's name on a new line.
""".strip()
    user_prompt = f"""
**User's Question:** "{user_question}"

**Survey Data Context:**
{context_for_prompt}

Begin the dialogue now.
""".strip()
    return {"system": system_prompt, "user": user_prompt}


def generate_dialogue(user_question: str, sources: List[Source]) -> str:
    prompts = _build_prompt(sources, user_question)
    response = client.chat.completions.create(
        model=DEFAULT_DIALOGUE_MODEL,
        messages=[
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompts["user"]},
        ],
        # temperature=0.7,
        # temperature not supported for this model
    )
    return response.choices[0].message.content.strip()


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/generate_dialogue", methods=["POST"])
def handle_generate_dialogue():
    payload = request.get_json(force=True, silent=True) or {}
    user_question = str(payload.get("question", "")).strip()
    if not user_question:
        return jsonify({"error": "Please provide a question."}), 400
    try:
        query_embedding = get_embedding(user_question)
        sources = find_most_relevant_responses(query_embedding)
        if not sources:
            return jsonify({"error": "No relevant survey responses found."}), 404
        dialogue_text = generate_dialogue(user_question, sources)
    except Exception as exc:  # noqa: BLE001 - surface full exception
        return jsonify({"error": str(exc)}), 500
    sources_payload = {str(source.id): source.__dict__ for source in sources}
    return jsonify({"dialogue": dialogue_text, "sources": sources_payload})


def _load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Processed data not found at {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


DATAFRAME = _load_data()
EMBEDDING_VECTORS = _prepare_embeddings(DATAFRAME)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
