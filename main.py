import os
from dotenv import load_dotenv
import google.generativeai as genai

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional

load_dotenv()

ENV = os.getenv("ENV", "PROD")
API_V1 = "/api/v1/ai"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Pydantic 모델 정의
class QuestionRequest(BaseModel):
    question: str
    max_tokens: Optional[int] = 1000

class AnswerResponse(BaseModel):
    question: str
    answer: str
    success: bool
    error: Optional[str] = None

# Gemini API 설정
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # model = genai.GenerativeModel('gemini-2.5-pro')
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
else:
    model = None

# start end 새로운 문법
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("서버 시작...")
    yield
    print("서버 종료...")

if ENV.upper() == "DEV":
    app = FastAPI(title="API", description="API for the project")
else:
    app = FastAPI(title="API", description="API for the project", docs_url=None, redoc_url=None)

# Gemini API 연동 함수
async def get_gemini_response(question: str) -> tuple[str, bool, str]:
    """
    Gemini API를 사용하여 질문에 대한 답변을 생성합니다.
    
    Returns:
        tuple: (답변, 성공여부, 에러메시지)
    """
    if not model:
        return "", False, "Gemini API 키가 설정되지 않았습니다."
    
    try:
        response = model.generate_content(question)
        if response.text:
            return response.text, True, None
        else:
            return "", False, "Gemini API에서 응답을 생성하지 못했습니다."
    except Exception as e:
        return "", False, f"Gemini API 호출 중 오류가 발생했습니다: {str(e)}"

@app.get(f"{API_V1}")
def read_root():
    return {"message": "AI 서버가 시작되었습니다."}

@app.post(f"{API_V1}/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Gemini API를 사용하여 질문에 답변합니다.
    """
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="Gemini API 키가 설정되지 않았습니다. 환경변수 GEMINI_API_KEY를 설정해주세요."
        )
    
    answer, success, error = await get_gemini_response(request.question)
    
    if not success:
        raise HTTPException(status_code=500, detail=error)
    
    return AnswerResponse(
        question=request.question,
        answer=answer,
        success=success,
        error=error
    )

# 예시 엔드포인트 (기존)
@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}