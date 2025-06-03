from typing import Optional
from sentence_transformers import SentenceTransformer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import httpx

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

async def search_similar_and_build_prompt(
    user_situation: str, user_thought: str, session: AsyncSession
) -> Optional[str]:
    query_text = f"{user_situation.strip()} {user_thought.strip()}"
    query_embedding = model.encode(query_text).tolist()
    vec_literal = "[" + ",".join(f"{x:.6f}" for x in query_embedding) + "]"

    sql = text("""
        SELECT
            r.situation,
            r.thought,
            r.reframe,
            d.trap_name,
            d.definition,
            d.tips,
            r.comb_embedding <-> CAST(:vec AS vector) AS distance
        FROM reframing_dataset r
        LEFT JOIN distortions d ON r.distortion_id = d.distortion_id
        ORDER BY r.comb_embedding <-> CAST(:vec AS vector)
        LIMIT 3;
    """)

    result = await session.execute(sql, {"vec": vec_literal})
    rows = result.mappings().all()
    if not rows:
        return None

    similar_section = ""
    for idx, row in enumerate(rows, start=1):
        similar_section += f"""
[Similar Case {idx}]
Situation: {row["situation"]}
Thought: {row["thought"]}
Reframed Thought: {row["reframe"]}
Distortion Type: {row["trap_name"] or "None"}
Definition: {row["definition"] or "N/A"}
Tips: {row["tips"] or "N/A"}
"""

    prompt = f"""
[User Situation]
{user_situation}

[User Thought]
{user_thought}

[Similar Examples]
{similar_section}

Please generate a response that explains the cognitive distortion pattern (if any) in the user's thought, using the examples above as reference. Also include possible reframing suggestions.
"""
    return prompt

async def ask_llm(prompt: str) -> str:
    payload = {
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    }
    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post("http://localhost:11434/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()
    return data.get("response", "")