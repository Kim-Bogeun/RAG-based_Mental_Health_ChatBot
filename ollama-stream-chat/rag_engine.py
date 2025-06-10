# rag_engine.py
from typing import Optional, List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import httpx

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# ──────────────────────────────────────────────
# 1. 예시 Thought + Distortion 메타데이터 검색
# ──────────────────────────────────────────────
async def fetch_top_k_similar_thoughts(
    user_embedding: List[float],
    session: AsyncSession,
    top_k: int = 3
) -> List[Dict]:
    """
    Return: [
        {
            'example_thought': str,
            'distortion_id'  : int,
            'trap_name'      : str,
            'definition'     : str,
            'tips'           : str
        }, …
    ]
    """
    sql = text("""
        SELECT
            r.thought                AS example_thought,
            r.distortion_id          AS distortion_id,
            d.trap_name,
            d.definition,
            d.tips,
            e.embedding <-> CAST(:vec AS vector) AS distance
        FROM example_embeddings AS e
        INNER JOIN example_dataset AS r
          ON e.embedding_id = r.embedding_id
        LEFT JOIN distortions AS d
          ON r.distortion_id = d.distortion_id
        ORDER BY distance
        LIMIT :k;
    """)

    result = await session.execute(
        sql,
        {
            "vec": "[" + ",".join(f"{x:.6f}" for x in user_embedding) + "]",
            "k":   top_k
        }
    )
    rows = result.mappings().all()
    return [
        {
            "example_thought": row["example_thought"],
            "distortion_id"  : row["distortion_id"],
            "trap_name"      : row["trap_name"] or "UnknownDistortion",
            "definition"     : row["definition"] or "Definition not available.",
            "tips"           : row["tips"] or "No tips available."
        }
        for row in rows
    ]

# ──────────────────────────────────────────────
# 2. distortion_id → Reframe 예시 n개(상황 + 생각 + 리프레임) 추출
# ──────────────────────────────────────────────
async def fetch_reframe_examples(
    distortion_id: int,
    session: AsyncSession,
    limit: int = 2
) -> List[Dict]:
    sql = text("""
        SELECT situation, thought, reframe
        FROM reframing_dataset
        WHERE distortion_id = :did
          AND reframe IS NOT NULL
        LIMIT :lim;
    """)
    result = await session.execute(sql, {"did": distortion_id, "lim": limit})
    rows = result.mappings().all()
    return [
        {
            "situation": row["situation"] or "(no situation provided)",
            "thought"  : row["thought"],
            "reframe"  : row["reframe"]
        }
        for row in rows
    ]

# ──────────────────────────────────────────────
# 3. 전체 프롬프트 생성 + 가장 유사한 distortion_id 반환
# ──────────────────────────────────────────────
async def search_similar_and_build_prompt(
    user_situation: str,
    user_thought  : str,
    session: AsyncSession,
    top_k: int = 3
) -> Optional[Tuple[str, int]]:
    """
    1) thought 임베딩 → example_embeddings에서 상위 k개 예시 찾기
    2) 각 distortion_id로 정의·팁·리프레이밍(상황+생각+예시) 조회
    3) “버전 1” 포맷에 맞춰 프롬프트 반환
    4) 가장 유사한 distortion_id도 함께 반환
    """

    # 1. 임베딩
    query_emb = model.encode(user_thought.strip()).tolist()

    # 2. 상위 k 예시 + 메타데이터
    similar_items = await fetch_top_k_similar_thoughts(query_emb, session, top_k)
    if not similar_items:
        return None

    top_distortion_id = similar_items[0]["distortion_id"]

    # 3. distortion_id → reframe 예시 캐시 (상황, 생각, 예시 리프레임)
    for item in similar_items:
        item["reframes"] = await fetch_reframe_examples(item["distortion_id"], session, limit=2)

    # 4. 프롬프트 조립
    prompt_parts: List[str] = []

    # ─── 머리말: Situation / Thought ───
    prompt_parts.append("[User Situation]\n" + (user_situation or "(none provided)") + "\n")
    prompt_parts.append("[User Thought]\n" + user_thought + "\n")

    # ─── Part 1: Candidate N ───
    prompt_parts.append(
        "1. Several possible cognitive distortions that may underlie the user's thoughts:\n"
    )
    for idx, it in enumerate(similar_items, 1):
        prompt_parts.append(
            f"Candidate {idx}: {it['trap_name']} "
            f"(Definition: {it['definition']})"
        )
    prompt_parts.append("")

    # ─── Part 2: Reframe N 블록 (few-shot guidance) ───
    prompt_parts.append(
        "2. Your task is to suggest alternative rational thoughts to address each of the identified cognitive distortions. Use the following tips and reframing examples:"
    )

    for idx, it in enumerate(similar_items, 1):
        prompt_parts.append(f"\n[Reframe {idx}]")
        prompt_parts.append(f"Tips to overcome {it['trap_name']}: {it['tips']}")

        for ridx, ex in enumerate(it["reframes"], 1):
            prompt_parts.append(f"Example Situation {ridx}: {ex['situation']}")
            prompt_parts.append(f"Example Original Thought   {ridx}: {ex['thought']}")
            prompt_parts.append(f"Example Reframed Thought   {ridx}: {ex['reframe']}")

        prompt_parts.append("")

    # ─── Part 3: 출력 포맷 지시 ───
    prompt_parts.append(
        "3. Now generate new reframed thoughts for the user's input (based on their Situation & Thought). Do not include the Example Situation and Example Thought in the output."
        "Output in the following format:\n"
        "[Notice] Reference only. For accurate evaluation of potential cognitive distortions, please consult a mental health professional.\n"
    )
    for idx in range(1, len(similar_items) + 1):
        prompt_parts.append(f"Candidate Distortion {idx}:")
        prompt_parts.append(f"Definition of the Distortion:")
        prompt_parts.append(f"Tips to Overcome the Distortion:")
        prompt_parts.append(f"Example Reframed Thoughts for the Distortion:\n")

    return "\n".join(prompt_parts), top_distortion_id

# ──────────────────────────────────────────────
# 4. LLM 호출
# ──────────────────────────────────────────────
async def ask_llm(prompt: str) -> str:
    payload = {
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    }
    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post("http://localhost:11434/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json().get("response", "")
