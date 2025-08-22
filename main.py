import os
import json
import re
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
class ChatMessage(BaseModel):
    speaker: str  # "user" 또는 "character"
    text: str

class NPCChatRequest(BaseModel):
    character_info: str
    background_situation: str
    previous_chat: list[ChatMessage]
    affinity: int
    last_user_input: str
    user_nickname: str
    user_gender: str

class NPCChatResponse(BaseModel):
    affinity: int
    next_utterance: str
    emotion: str

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

# NPC 채팅 로직
async def get_npc_chat_response(request: NPCChatRequest) -> tuple[NPCChatResponse, bool, str]:
    """
    Gemini API를 사용하여 NPC 채팅 응답을 생성합니다.
    
    Returns:
        tuple: (NPC응답, 성공여부, 에러메시지)
    """
    if not model:
        return None, False, "Gemini API 키가 설정되지 않았습니다."
    
    try:
        # 채팅 히스토리 구성
        chat_history = ""
        for chat in request.previous_chat:
            if chat.speaker == "user":
                chat_history += f"{request.user_nickname}: {chat.text}\n"
            else:
                chat_history += f"NPC: {chat.text}\n"
        
        # 프롬프트 구성
        prompt = f"""You are roleplaying as an NPC character in a story. Generate your response based on the given context.

CHARACTER: {request.character_info}

SETTING: {request.background_situation}

CURRENT AFFINITY: {request.affinity}/100

PLAYER: {request.user_nickname} ({request.user_gender})

CONVERSATION HISTORY:
{chat_history}

PLAYER'S LATEST MESSAGE: {request.last_user_input}

IMPORTANT: You must respond with ONLY a valid JSON object. No other text before or after. Use this exact format:

{{"affinity": number, "next_utterance": "text", "emotion": "emotion"}}

Where:
- affinity: integer from 0-100 (adjust based on player's message, usually ±1 to ±5)
- next_utterance: your character's response as a string
- emotion: exactly one of these: "Neutral", "Happiness", "Sadness", "Feel_affection", "Anger"

Example: {{"affinity": 45, "next_utterance": "I see you have some confidence.", "emotion": "Neutral"}}

Respond only with the JSON object:"""
        response = model.generate_content(prompt)
        if response.text:
            # JSON 응답 파싱
            try:
                response_text = response.text.strip()
                print(f"Raw Gemini response: {response_text}")
                
                # 다양한 형태의 마크다운 코드 블록 제거
                if response_text.startswith('```json'):
                    response_text = response_text[7:].strip()
                    if response_text.endswith('```'):
                        response_text = response_text[:-3].strip()
                elif response_text.startswith('```'):
                    response_text = response_text[3:].strip()
                    if response_text.endswith('```'):
                        response_text = response_text[:-3].strip()
                
                # JSON 객체 부분만 추출 (중괄호로 시작하고 끝나는 부분)
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(0)
                
                print(f"Cleaned response text: {response_text}")
                
                response_data = json.loads(response_text)
                
                # 필수 필드 검증
                if not all(key in response_data for key in ["affinity", "next_utterance", "emotion"]):
                    missing_keys = [key for key in ["affinity", "next_utterance", "emotion"] if key not in response_data]
                    return None, False, f"AI 응답에 필수 필드가 누락되었습니다: {missing_keys}"
                
                # 값 검증 및 기본값 설정
                affinity = int(response_data.get("affinity", request.affinity))
                affinity = max(0, min(100, affinity))  # 0-100 범위 제한
                
                emotion = response_data.get("emotion", "Neutral")
                valid_emotions = ["Neutral", "Happiness", "Sadness", "Feel_affection", "Anger"]
                if emotion not in valid_emotions:
                    emotion = "Neutral"
                
                npc_response = NPCChatResponse(
                    affinity=affinity,
                    next_utterance=str(response_data.get("next_utterance", "...")),
                    emotion=emotion
                )
                
                return npc_response, True, None
                
            except json.JSONDecodeError as e:
                return None, False, f"AI 응답을 파싱하는 중 오류가 발생했습니다. 원본 응답: '{response_text}', 오류: {str(e)}"
            except (ValueError, TypeError) as e:
                return None, False, f"AI 응답 데이터 형식이 올바르지 않습니다: {str(e)}"
        else:
            return None, False, "Gemini API에서 응답을 생성하지 못했습니다."
    except Exception as e:
        return None, False, f"Gemini API 호출 중 오류가 발생했습니다: {str(e)}"

@app.get(f"{API_V1}")
def read_root():
    return {"message": "AI 서버가 시작되었습니다."}

@app.post(f"{API_V1}/chat", response_model=NPCChatResponse)
async def npc_chat(request: NPCChatRequest):
    """
    NPC와의 채팅 결과를 반환하는 API입니다.
    """
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="Gemini API 키가 설정되지 않았습니다. 환경변수 GEMINI_API_KEY를 설정해주세요."
        )
    
    npc_response, success, error = await get_npc_chat_response(request)
    
    if not success:
        raise HTTPException(status_code=500, detail=error)
    
    return npc_response

# 예시 엔드포인트 (기존)
@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}