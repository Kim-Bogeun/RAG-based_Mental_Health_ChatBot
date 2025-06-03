import streamlit as st
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from rag_engine import search_similar_and_build_prompt, ask_llm
from database import async_session, init_db

# 페이지 설정
st.set_page_config(page_title="Cognitive Distortion Chatbot", layout="wide")
st.title("🧠 Cognitive Reframing Assistant")

# 입력창
user_situation = st.text_area("Describe the situation", height=100)
user_thought = st.text_area("What thought came to your mind?", height=100)

# 버튼 누를 때 실행
if st.button("Analyze Thought"):
    if not user_situation.strip() or not user_thought.strip():
        st.warning("Please provide both a situation and a thought.")
    else:
        async def main():
            await init_db()
            async with async_session() as session:
                with st.spinner("Retrieving similar cases and generating explanation..."):
                    prompt = await search_similar_and_build_prompt(user_situation, user_thought, session)
                    if not prompt:
                        st.error("❌ No relevant examples found.")
                        return

                    response = await ask_llm(prompt)
                    st.subheader("🧾 Generated Explanation")
                    st.markdown(response)
                    with st.expander("📄 Prompt Sent to LLM"):
                        st.code(prompt)

        asyncio.run(main())