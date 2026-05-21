"""
llm.py — LLM wrappers and prompt templates for query parsing and answer generation.
"""

from langchain_openai import ChatOpenAI
from src.config import config


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=config.OPENAI_CHAT_MODEL,
        api_key=config.OPENAI_API_KEY,
        temperature=0,
    )


def generate_answer(query: str, contexts: list[str]) -> str:
    llm = get_llm()
    joined_context = "\n\n---\n\n".join(contexts)

    prompt = f"""You are a clinical data analysis assistant.

Your task is to answer the user's question using ONLY the provided clinical note context.
Do NOT use outside medical knowledge.
Do NOT guess or fabricate evidence.
If the answer cannot be supported directly by the retrieved context, say:
"Unable to confirm from the available retrieved notes."

Before writing the final response, identify the strongest supporting note(s) from
the retrieved context, including note_id/source when available. Use that evidence to
decide whether the answer is supported.

Do NOT include an Evidence section in the final response — the application shows
retrieved notes separately in the Evidence Vault.

Format your response exactly as:

Answer:
<detailed answer based only on the retrieved context; usually 4–8 sentences>

Confidence:
<High / Medium / Low>

Reason:
<brief reason based on the number, specificity, and consistency of supporting notes>

User question:
{query}

Retrieved context:
{joined_context}
"""
    return llm.invoke(prompt).content.strip()


def fallback_answer(query: str) -> str:
    llm = get_llm()

    prompt = f"""You are a clinical data analysis assistant.

The retrieval pipeline could not find strong supporting evidence for the user's question.
Respond conservatively. Do NOT fabricate facts or rely on outside medical knowledge.

Format your response exactly as:

Answer:
Unable to confirm from the available retrieved notes.

Confidence:
Low

Reason:
Retrieval returned weak or insufficient supporting evidence.

User question:
{query}
"""
    return llm.invoke(prompt).content.strip()


def summarize_patient_notes(context: str) -> str:
    llm = get_llm()

    prompt = f"""You are assisting a physician.

These are multiple notes for one patient. Create a concise patient summary covering:
- Major diagnoses and chronic conditions
- Relevant psychiatric / social history
- Important admissions or presentations
- Treatments and procedures
- Clinical trajectory over time
- Current risks or follow-up needs if mentioned

Be concise and clinically useful.
Do NOT fabricate information not explicitly supported by the notes.

Format your response exactly as:

Answer:
<patient summary>

Confidence:
<High / Medium / Low>

Reason:
<brief explanation based on note consistency and completeness>

Notes:
{context}
"""
    return llm.invoke(prompt).content.strip()


def summarize_single_note(context: str) -> str:
    llm = get_llm()

    prompt = f"""You are assisting a physician. Summarize this clinical note clearly.

Include:
- Why the patient presented
- Important findings
- Treatments and procedures
- Key diagnoses
- Disposition and follow-up

Do NOT fabricate information not explicitly supported by the note.

Format your response exactly as:

Answer:
<clinical summary>

Confidence:
<High / Medium / Low>

Reason:
<brief explanation based on note specificity and completeness>

Note:
{context}
"""
    return llm.invoke(prompt).content.strip()


def summarize_visit_notes(context: str) -> str:
    llm = get_llm()

    prompt = f"""You are assisting a physician.

These notes belong to one hospitalization. Summarize:
- Reason for admission
- Hospital course
- Procedures and treatments
- Key findings
- Discharge condition
- Follow-up needs

Do NOT fabricate information not explicitly supported by the notes.

Format your response exactly as:

Answer:
<hospitalization summary>

Confidence:
<High / Medium / Low>

Reason:
<brief explanation based on consistency and coverage of the hospitalization notes>

Notes:
{context}
"""
    return llm.invoke(prompt).content.strip()
