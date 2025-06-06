# streamlit_app.py

import os
import asyncio

# ── Streamlit이 torch 내부를 감시하지 않도록 반드시 streamlit import 전에 설정 ──
os.environ["STREAMLIT_WATCH_SKIP_PACKAGES"] = "torch"

import streamlit as st
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from rag_engine import search_similar_and_build_prompt, ask_llm
from database import async_session, init_db

# 페이지 설정
st.set_page_config(page_title="Cognitive Distortion Chatbot", layout="wide")
st.title("🧠 Cognitive Reframing Assistant")

user_situation = st.text_area("Describe the situation", height=70)
user_thought   = st.text_area("What thought came to your mind?", height=70)
user_id        = st.text_input("Please input your ID (for future use)", value="")

if st.button("Analyze Thought"):
    if not user_situation.strip() or not user_thought.strip() or not user_id.strip():
        st.warning("Please provide situation, thought, and your ID.")
        st.stop()

    async def main():
        # 1) DB 초기화 (최초 1회만 해도 된다면 생략 가능)
        await init_db()

        # 2) AsyncSession 열기
        async with async_session() as session:
            with st.spinner("Retrieving similar cases and generating explanation..."):
                # (2.1) RAG 프롬프트 생성
                prompt = await search_similar_and_build_prompt(
                    user_situation,
                    user_thought,
                    session
                )
                if not prompt:
                    st.error("❌ No relevant examples found.")
                    return

                # (2.2) LLM 호출
                answer = await ask_llm(prompt)

                # ── 여기부터 수정 구간 ──

                # (2.3-1) users 테이블에 user_id가 없으면 추가 (ON CONFLICT DO NOTHING)
                await session.execute(
                    text("""
                        INSERT INTO users (user_id)
                        VALUES (:uid)
                        ON CONFLICT (user_id) DO NOTHING;
                    """),
                    {"uid": user_id}
                )

                # (2.3-2) logs 테이블에 삽입
                await session.execute(
                    text("""
                        INSERT INTO logs (user_id, query, answer)
                        VALUES (:uid, :q, :a);
                    """),
                    {
                        "uid": user_id,
                        "q"  : f"Situation: {user_situation} | Thought: {user_thought}",
                        "a"  : answer
                    }
                )

                # (2.4) 커밋
                await session.commit()
                # ── 수정 구간 끝 ──

        # (3) 결과 출력
        st.subheader("🧾 Generated Explanation")
        st.markdown(answer)
        with st.expander("📄 Prompt Sent to LLM"):
            st.code(prompt)

    # (4) asyncio.run(...)으로 비동기 함수 실행
    asyncio.run(main())