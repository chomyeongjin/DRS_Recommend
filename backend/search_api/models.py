from pydantic import BaseModel, Field
from typing import List

class IngestRequest(BaseModel):
    days: int = 1825  # 5년 (데이터 수집 기본값)

class SketchRequest(BaseModel):
    y: List[float] = Field(..., min_items=10)  # 스케치 y값 (0~1 권장)
    target_len: int = 200
    period: str = "1y"  # "3m", "1y", "2y"

class CompareTickerRequest(SketchRequest):
    ticker: str

class SimilarResponseItem(BaseModel):
    ticker: str
    name: str = ""  # 회사 이름
    score: float
    rank: int
    series_norm: List[float]
    sketch_norm: List[float]

class SimilarResponse(BaseModel):
    items: List[SimilarResponseItem]