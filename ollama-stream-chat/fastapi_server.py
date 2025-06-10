from fastapi import FastAPI, APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from rag_engine import (
    search_similar_and_build_prompt,
    ask_llm
)

# === DB 연결 설정 ===
DB_URL = "postgresql+asyncpg://jangjiwon@localhost:5432/cognitive_distortion"
engine = create_async_engine(DB_URL, echo=False)
async_session = async_sessionmaker(engine, expire_on_commit=False)

# === FastAPI 앱, 라우터 선언 ===
app = FastAPI(title="Cognitive Distortion Explanation RAG API")
router = APIRouter()

# === Pydantic 모델 ===
class ExplanationQuery(BaseModel):
    situation: str = Field(..., example="시험에서 떨어졌어요.")
    thought: str = Field(..., example="나는 항상 실패하는 사람 같아.")

class ExplanationResponse(BaseModel):
    response: str
    prompt: Optional[str]
    has_info: bool

# === 의존성: DB 세션 제공 ===
async def get_db_session() -> AsyncSession:
    async with async_session() as session:
        yield session

# === 핵심 POST API 라우트 ===
@router.post("/query_explanation", response_model=ExplanationResponse)
async def query_explanation(
    req: ExplanationQuery,
    session: AsyncSession = Depends(get_db_session)
):
    try:
        prompt = await search_similar_and_build_prompt(
            req.thought,
            session
        )
        if not prompt:
            return ExplanationResponse(
                response="⚠️ No relevant counseling information was found.",
                prompt=None,
                has_info=False
            )
        answer = await ask_llm(prompt)
        return ExplanationResponse(
            response=answer,
            prompt=prompt,
            has_info=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    
# === 라우터 등록 ===
app.include_router(router, prefix="/api")

# === 헬스체크용 GET 엔드포인트 ===
@app.get("/api/health")
async def health_check():
    return {"status": "ok"}