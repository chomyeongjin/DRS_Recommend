from google import genai
from pydantic import BaseModel
import logging
from typing import List

from .config import settings

logger = logging.getLogger(__name__)

class PatternAnalysis(BaseModel):
    pattern_name: str
    fun_fact: str

def analyze_sketch_pattern(sketch_y: List[float]) -> dict:
    """
    Gemini API를 사용하여 스케치 y좌표를 분석하고 주식 패턴 이름과 Fun Fact를 반환
    """
    if not settings.gemini_api_key:
        logger.warning("Gemini API key is missing. Skipping AI analysis.")
        return {
            "pattern_name": "Unknown Pattern (API Key Missing)",
            "fun_fact": "Gemini API 키를 설정하면 AI 분석을 시작할 수 있습니다."
        }
        
    try:
        client = genai.Client(api_key=settings.gemini_api_key)
        
        prompt = (
            "You are an expert stock market technical analyst. "
            "I will provide you with a list of 128 normalized Y-coordinates (from 0 to 1) representing a stock chart sketch drawn by a user. "
            "Examine the shape of this curve and classify it as one of the classic stock chart patterns (e.g., Head and Shoulders, Inverse Head and Shoulders, Double Top, Double Bottom, Cup and Handle, Triangle, Rising Wedge, Falling Wedge, Triple Top, Triple Bottom, etc.).\n"
            "If it doesn't clearly match any, use your best judgment to pick the closest one or describe the general trend (e.g., Uptrend, Downtrend, Range Bound).\n\n"
            "Return the identified pattern name (translated reasonably well into Korean if applicable, e.g., '헤드 앤 숄더', '더블 바텀 (쌍바닥)', '상승 박스권' etc., or just keeping it clear) "
            "and a 'fun fact' or basic explanation about this pattern (also in Korean).\n\n"
            f"Here are the Y-coordinates of the sketch: {sketch_y}"
        )

        response = client.models.generate_content(
            model='gemini-flash-lite-latest',
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': PatternAnalysis,
            },
        )
        
        if response.text:
            import json
            try:
                return json.loads(response.text)
            except Exception:
                return {"pattern_name": "Parsing Error", "fun_fact": "Failed to parse JSON response."}
        else:
            return {"pattern_name": "Analysis Failed", "fun_fact": "Could not parse Gemini's response."}
            
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        return {
            "pattern_name": "API Error",
            "fun_fact": f"Error interacting with the AI model: {str(e)}"
        }
