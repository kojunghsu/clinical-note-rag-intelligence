import json
from src.llm import get_llm


SCHEMA = """
{
  "intent": "patient_summary | note_summary | visit_summary | retrieval_search",
  "subject_id": null,
  "note_id": null,
  "hadm_id": null,

  "history": [],
  "medications": [],
  "results": [],
  "hospital_course": [],
  "discharge": [],
  "outcomes": false,

  "semantic_search": ""
}
"""


def parse_query(query: str) -> dict:
    llm = get_llm()

    prompt = f"""
You are a medical query parser.

Return ONLY valid JSON matching this schema:

{SCHEMA}

Rules:
- If the user asks about a patient/subject id, use intent = "patient_summary"
- If the user asks about a note id, use intent = "note_summary"
- If the user asks about a hadm_id / admission / hospitalization id, use intent = "visit_summary"
- For every other non-ID query, use intent = "retrieval_search"
- Extract IDs if present
- Put remaining free-text request in semantic_search
- No explanation, no markdown, JSON only

Query:
{query}
"""

    print("[PARSER] Sending query to LLM parser...")
    raw = llm.invoke(prompt).content.strip()
    print("[PARSER] Raw JSON returned:")
    print(raw)

    try:
        parsed = json.loads(raw)
        print("[PARSER] Parse successful")
        return parsed
    except Exception:
        return {
            "intent": "retrieval_search",
            "subject_id": None,
            "note_id": None,
            "hadm_id": None,
            "history": [],
            "medications": [],
            "results": [],
            "hospital_course": [],
            "discharge": [],
            "outcomes": False,
            "semantic_search": query
        }