import os
import json
import re
from dotenv import load_dotenv
import google.generativeai as genai

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional, Annotated
from typing_extensions import TypedDict

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from IPython.display import Image, display

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
    isGrammarError: bool
    last_user_input: str
    last_user_correction_input: str

# LangGraph State 정의
class NPCChatState(TypedDict):
    """NPC 채팅을 위한 LangGraph State"""
    request: NPCChatRequest
    chat_history: str
    grammar_check: str
    has_grammar_error: bool
    corrected_sentence: str
    prompt: str
    raw_response: str
    response: NPCChatResponse
    error: Optional[str]

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

# LangGraph 노드 함수들
def prepare_chat_history(state: NPCChatState) -> NPCChatState:
    """이전 대화 기록을 정리하는 노드"""
    request = state["request"]
    chat_history = ""
    
    for chat in request.previous_chat:
        if chat.speaker == "user":
            chat_history += f"{request.user_nickname}: {chat.text}\n"
        else:
            chat_history += f"NPC: {chat.text}\n"
    
    return {**state, "chat_history": chat_history}

def check_grammar(state: NPCChatState) -> NPCChatState:
    """Check English grammar of user input and correct if needed"""
    if not model:
        return {**state, "error": "Gemini API 키가 설정되지 않았습니다."}
    
    try:
        request = state["request"]
        
        # 1단계: 문법 검사
        grammar_check_prompt = f"""Please determine if the following text has correct English grammar by checking three key grammar elements. Answer with only 'correct' or 'incorrect'.

Text: "{request.last_user_input}"

Check these three core grammar elements:

1. Tense (Verb Tenses)
   - Correct examples: "I went to school yesterday", "I will go tomorrow", "I am going now"
   - Incorrect examples: "I go to school yesterday" (present with past time), "I will went tomorrow" (future + past form), "I am went now" (continuous + past form)

2. Word Order (Sentence Structure)
   - Correct examples: "I like apples", "She is reading a book", "We went to the store"
   - Incorrect examples: "Like I apples" (verb before subject), "Reading she is a book" (scrambled order), "Went we to store the" (wrong word positioning)

3. Articles (A, An, The)
   - Correct examples: "I saw a cat", "The book is good", "An apple is red"
   - Incorrect examples: "I saw cat" (missing article), "A book is good" (should be 'the' for specific), "A apple is red" (wrong article before vowel)

Review all three criteria. If ANY element is incorrect, answer 'incorrect'. Only if ALL are correct, answer 'correct'.

Answer: (write only 'correct' or 'incorrect')"""
        
        response = model.generate_content(grammar_check_prompt)
        
        if not response.text:
            return {**state, "error": "문법 검사 중 응답을 받지 못했습니다."}
        
        grammar_result = response.text.strip().lower()
        
        # 2단계: 문법 오류가 있는 경우 수정
        if "incorrect" in grammar_result:
            correction_prompt = f"""Please correct the following English sentence that has grammar errors. Focus on fixing tense, word order, and article issues. Return only the corrected sentence, nothing else.

Original sentence: "{request.last_user_input}"

Corrected sentence:"""
            
            correction_response = model.generate_content(correction_prompt)
            
            if correction_response.text:
                corrected_sentence = correction_response.text.strip()
                return {
                    **state, 
                    "grammar_check": grammar_result,
                    "has_grammar_error": True,
                    "corrected_sentence": corrected_sentence
                }
            else:
                return {
                    **state, 
                    "grammar_check": grammar_result,
                    "has_grammar_error": True,
                    "corrected_sentence": request.last_user_input  # 수정 실패시 원문 유지
                }
        else:
            # 문법이 올바른 경우
            return {
                **state, 
                "grammar_check": grammar_result,
                "has_grammar_error": False,
                "corrected_sentence": request.last_user_input
            }
            
    except Exception as e:
        return {**state, "error": f"문법 검사 중 오류가 발생했습니다: {str(e)}"}



def generate_normal_prompt(state: NPCChatState) -> NPCChatState:
    """Generate prompt that handles both normal conversation and grammar correction"""
    request = state["request"]
    chat_history = state["chat_history"]
    has_grammar_error = state.get("has_grammar_error", False)
    corrected_sentence = state.get("corrected_sentence", request.last_user_input)
    
    if has_grammar_error:
        # 문법 오류가 있는 경우의 프롬프트
        prompt = f"""You are roleplaying as an NPC character in a story. The user made a grammatical error in their message, but you understand what they meant. Gently correct them while staying in character and continuing the conversation naturally.

CHARACTER: {request.character_info}

SETTING: {request.background_situation}

CURRENT AFFINITY: {request.affinity}/100

PLAYER: {request.user_nickname} ({request.user_gender})

CONVERSATION HISTORY:
{chat_history}

PLAYER'S ORIGINAL MESSAGE (with grammar error): {request.last_user_input}
CORRECTED VERSION: {corrected_sentence}

IMPORTANT GUIDELINES:
1. Stay completely in character - do not break the roleplay
2. Gently correct the grammar naturally as part of your response (e.g., "You mean to say..." or "I think you're asking..." or simply repeat the correct version)
3. The correction should feel like a natural part of the conversation flow, not like a grammar lesson
4. Do not be preachy or make the user feel bad about the mistake
5. Continue the conversation naturally after the gentle correction
6. The affinity should decrease slightly (1-2 points) due to the correction
7. Make the correction sound helpful and friendly, not condescending

You must respond with ONLY a valid JSON object. No other text before or after. Use this exact format:

{{"affinity": number, "next_utterance": "text", "emotion": "emotion"}}

Where:
- affinity: integer from 0-100 (decrease by 1-2 points due to gentle correction)
- next_utterance: your character's response that naturally corrects the grammar while staying in character
- emotion: exactly one of these: "Neutral", "Happiness", "Sadness", "Feel_affection", "Anger"

Example: {{"affinity": 60, "next_utterance": "I think you meant 'How is business today?' Well, business isn't going great, but I appreciate polite customers like you. Maybe you'd be interested in today's special offer?", "emotion": "Neutral"}}

Respond only with the JSON object:"""
    else:
        # 문법이 올바른 경우의 일반 프롬프트
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
- affinity: integer from 0-100 (adjust based on player's message, changes by one of -3, -2, -1, +1, +2, or +3)
- next_utterance: your character's response as a string
- emotion: exactly one of these: "Neutral", "Happiness", "Sadness", "Feel_affection", "Anger"

Example: {{"affinity": 45, "next_utterance": "I see you have some confidence.", "emotion": "Neutral"}}

Respond only with the JSON object:"""
    
    return {**state, "prompt": prompt}



def generate_response(state: NPCChatState) -> NPCChatState:
    """Gemini API를 사용하여 응답을 생성하는 노드"""
    if not model:
        return {**state, "error": "Gemini API 키가 설정되지 않았습니다."}
    
    try:
        prompt = state["prompt"]
        response = model.generate_content(prompt)
        
        if response.text:
            return {**state, "raw_response": response.text}
        else:
            return {**state, "error": "Gemini API에서 응답을 생성하지 못했습니다."}
    except Exception as e:
        return {**state, "error": f"Gemini API 호출 중 오류가 발생했습니다: {str(e)}"}

def parse_response(state: NPCChatState) -> NPCChatState:
    """응답을 파싱하여 NPCChatResponse 객체로 변환하는 노드"""
    if "error" in state and state["error"]:
        return state
    
    try:
        response_text = state["raw_response"].strip()
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
            return {**state, "error": f"AI 응답에 필수 필드가 누락되었습니다: {missing_keys}"}
        
        # 값 검증 및 기본값 설정
        request = state["request"]
        affinity = int(response_data.get("affinity", request.affinity))
        affinity = max(0, min(100, affinity))  # 0-100 범위 제한
        
        emotion = response_data.get("emotion", "Neutral")
        valid_emotions = ["Neutral", "Happiness", "Sadness", "Feel_affection", "Anger"]
        if emotion not in valid_emotions:
            emotion = "Neutral"
        
        # State에서 문법 관련 정보 추출
        has_grammar_error = state.get("has_grammar_error", False)
        corrected_sentence = state.get("corrected_sentence", "")
        
        npc_response = NPCChatResponse(
            affinity=affinity,
            next_utterance=str(response_data.get("next_utterance", "...")),
            emotion=emotion,
            isGrammarError=has_grammar_error,
            last_user_input=request.last_user_input,
            last_user_correction_input=corrected_sentence if has_grammar_error else ""
        )
        
        return {**state, "response": npc_response}
        
    except json.JSONDecodeError as e:
        return {**state, "error": f"AI 응답을 파싱하는 중 오류가 발생했습니다. 원본 응답: '{response_text}', 오류: {str(e)}"}
    except (ValueError, TypeError) as e:
        return {**state, "error": f"AI 응답 데이터 형식이 올바르지 않습니다: {str(e)}"}

# LangGraph 워크플로우 생성
def create_npc_chat_graph():
    """NPC 채팅을 위한 LangGraph 워크플로우 생성"""
    graph_builder = StateGraph(NPCChatState)
    
    # 노드 추가
    graph_builder.add_node("prepare_history", prepare_chat_history)
    graph_builder.add_node("check_grammar", check_grammar)
    graph_builder.add_node("generate_normal_prompt", generate_normal_prompt)
    graph_builder.add_node("generate_response", generate_response)
    graph_builder.add_node("parse_response", parse_response)
    
    # 엣지 연결: 단순한 선형 구조
    graph_builder.add_edge(START, "prepare_history")
    graph_builder.add_edge("prepare_history", "check_grammar")
    graph_builder.add_edge("check_grammar", "generate_normal_prompt")
    graph_builder.add_edge("generate_normal_prompt", "generate_response")
    graph_builder.add_edge("generate_response", "parse_response")
    graph_builder.add_edge("parse_response", END)
    
    return graph_builder.compile()

# 글로벌 그래프 인스턴스
npc_chat_graph = create_npc_chat_graph()

# NPC 채팅 로직 (LangGraph 사용)
async def get_npc_chat_response(request: NPCChatRequest) -> tuple[NPCChatResponse, bool, str]:
    """
    Generate NPC chat response using LangGraph with English grammar checking.
    
    Returns:
        tuple: (NPC response, success status, error message)
    """
    try:
        # Create initial state
        initial_state: NPCChatState = {
            "request": request,
            "chat_history": "",
            "grammar_check": "",
            "has_grammar_error": False,
            "corrected_sentence": "",
            "prompt": "",
            "raw_response": "",
            "response": None,
            "error": None
        }
        
        # LangGraph 실행
        final_state = npc_chat_graph.invoke(initial_state)
        
        # 결과 처리
        if "error" in final_state and final_state["error"]:
            return None, False, final_state["error"]
        
        if "response" in final_state and final_state["response"]:
            return final_state["response"], True, None
        else:
            return None, False, "Failed to generate response."
            
    except Exception as e:
        return None, False, f"Error occurred during LangGraph execution: {str(e)}"

@app.get(f"{API_V1}")
def read_root():
    return {"message": "AI 서버가 시작되었습니다."}

@app.get(f"{API_V1}/graph")
def get_graph_visualization():
    """Returns LangGraph workflow visualization information for English conversation NPC chat."""
    try:
        # Mermaid 형식으로 그래프 구조 반환
        mermaid_str = """
graph TD
    START([Start]) --> prepare_history[Prepare Chat History]
    prepare_history --> check_grammar[Check & Correct Grammar]
    check_grammar --> generate_normal_prompt[Generate Unified Prompt]
    generate_normal_prompt --> generate_response[Gemini API Call]
    generate_response --> parse_response[Parse Response]
    parse_response --> END([End])
"""
        return {
            "workflow": "NPC Chat LangGraph Workflow (Simplified Structure)",
            "nodes": [
                {"name": "prepare_history", "description": "Prepare previous conversation history"},
                {"name": "check_grammar", "description": "Check English grammar and correct if needed"},
                {"name": "generate_normal_prompt", "description": "Generate appropriate prompt based on grammar status"},
                {"name": "generate_response", "description": "Generate response using Gemini API"},
                {"name": "parse_response", "description": "Parse and validate JSON response"}
            ],
            "features": [
                "Automatic grammar error detection and correction",
                "Single prompt node handles both normal and correction scenarios",
                "Simplified linear workflow structure"
            ],
            "description": "This simplified workflow checks English grammar, corrects errors if found, and generates appropriate NPC responses. Grammar correction is handled seamlessly within the unified prompt generation.",
            "mermaid": mermaid_str.strip()
        }
    except Exception as e:
        return {"error": f"Error generating graph visualization: {str(e)}"}

def save_graph_image(filename: str = "npc_chat_workflow.png"):
    """
    LangGraph 워크플로우를 이미지 파일로 저장합니다.
    
    Args:
        filename: 저장할 파일명 (기본값: "npc_chat_workflow.png")
    
    Returns:
        bool: 저장 성공 여부
    """
    try:
        # LangGraph의 내장 시각화 기능 사용
        graph_image = npc_chat_graph.get_graph().draw_mermaid_png()
        
        # 이미지를 파일로 저장
        with open(filename, "wb") as f:
            f.write(graph_image)
        
        print(f"✅ NPC 채팅 LangGraph 워크플로우 이미지가 저장되었습니다: {filename}")
        
        return True
        
    except Exception as e:
        print(f"❌ 그래프 이미지 저장 중 오류가 발생했습니다: {str(e)}")
        return False

save_graph_image()

@app.post(f"{API_V1}/chat", response_model=NPCChatResponse)
async def npc_chat(request: NPCChatRequest):
    """
    API that returns NPC chat results with English grammar checking.
    Automatically corrects grammar errors while maintaining character roleplay.
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