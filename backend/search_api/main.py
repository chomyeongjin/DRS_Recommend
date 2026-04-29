# app/main.py
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import numpy as np, pandas as pd, time
import logging
import threading
from pathlib import Path

from .config import settings
from .models import IngestRequest, SketchRequest, SimilarResponse, SimilarResponseItem
from .tickers import get_tickers, get_ticker_info
from .data_io import (
    download_ohlc,  # fallback
    last_n_days, compute_ma20,
    save_ma20_parquet, save_meta, load_ma20_parquet
)
from .features import dict_to_matrix, normalize_pipeline
from .similar import rank_top_k
from .ai_analyzer import analyze_sketch_pattern
from . import db_io

# Logging configuration
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# FastAPI app
app = FastAPI(title=settings.api_title)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware with restricted origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"]
)

# Mount static files (web frontend)
web_dir = Path(__file__).parent.parent / "web"
if web_dir.exists():
    app.mount("/web", StaticFiles(directory=str(web_dir)), name="web")
    logger.info(f"Mounted static files from {web_dir}")

# Global cache with thread safety
CACHE = {
    "matrix": None,
    "tickers": None,
    "target_len": settings.target_len,
    "norm_map": None,   # ticker -> normalized vector(list)
    "ticker_info": None,  # ticker -> company name mapping
}
CACHE_LOCK = threading.Lock()

@app.on_event("startup")
def warmup():
    """서버 시작 시 기존 캐시(parquet)가 있으면 메모리 캐시 생성, DB 연결 초기화"""
    logger.info("Starting warmup: loading cached data...")

    # Load ticker info (symbol -> company name mapping)
    try:
        ticker_info = get_ticker_info()
        with CACHE_LOCK:
            CACHE["ticker_info"] = ticker_info
        logger.info(f"Loaded ticker info for {len(ticker_info)} companies")
    except Exception as e:
        logger.warning(f"Failed to load ticker info: {e}")
        CACHE["ticker_info"] = {}

    # Initialize PostgreSQL connection pool if data_source is postgresql
    if settings.data_source == "postgresql":
        try:
            db_io.init_pool(
                host=settings.pg_host,
                port=settings.pg_port,
                database=settings.pg_database,
                user=settings.pg_user,
                password=settings.pg_password,
                minconn=settings.pg_min_conn,
                maxconn=settings.pg_max_conn
            )
            seg_count = db_io.get_segment_count()
            logger.info(f"PostgreSQL connection initialized: {seg_count} segments available")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL connection: {e}")

    # Load parquet cache for backward compatibility
    try:
        df = load_ma20_parquet()
        if df is not None and not df.empty:
            # Check if new format (with 'ticker' and 'vector' columns)
            if 'ticker' in df.columns and 'vector' in df.columns:
                # New format: pre-computed vectors
                import numpy as np
                T = df['ticker'].tolist()
                # Convert vector column to matrix
                vectors = []
                for vec in df['vector']:
                    if isinstance(vec, np.ndarray):
                        vectors.append(vec)
                    elif isinstance(vec, list):
                        vectors.append(np.array(vec))
                    else:
                        # Skip invalid vectors
                        continue

                if vectors:
                    matrix = np.vstack(vectors)
                    norm_map = {t: matrix[i, :].tolist() for i, t in enumerate(T)}

                    with CACHE_LOCK:
                        CACHE.update({"matrix": matrix, "tickers": T, "norm_map": norm_map})

                    logger.info(f"Warmup completed: {len(T)} tickers loaded from pre-computed vectors")
                else:
                    logger.warning("No valid vectors found in parquet cache")
            else:
                # Old format: MA20 time series
                ma20 = {c: df[c].dropna() for c in df.columns}
                matrix, T = dict_to_matrix(ma20, target_len=CACHE["target_len"])
                norm_map = {t: matrix[i, :].tolist() for i, t in enumerate(T)}

                with CACHE_LOCK:
                    CACHE.update({"matrix": matrix, "tickers": T, "norm_map": norm_map})

                logger.info(f"Warmup completed: {len(T)} tickers loaded into cache")
        else:
            logger.info("No cached data found. Please run /ingest first.")
    except Exception as e:
        logger.error(f"Warmup failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/stats")
def stats():
    """현재 캐시된 티커 개수와 데이터 소스 정보 반환"""
    ticker_count = len(CACHE.get("tickers", [])) if CACHE.get("tickers") else 0

    # PostgreSQL 세그먼트 개수 (data_source가 postgresql인 경우)
    segment_count = 0
    if settings.data_source == "postgresql":
        try:
            segment_count = db_io.get_segment_count()
        except:
            pass

    return {
        "ticker_count": ticker_count,
        "segment_count": segment_count,
        "data_source": settings.data_source,
        "target_len": CACHE["target_len"]
    }

@app.post("/ingest")
@limiter.limit(settings.rate_limit_ingest)
def ingest(
    request: Request,
    req: IngestRequest,
    force_refresh: bool = Query(False),
    max_tickers: int = Query(5000, ge=10, le=5000)
):
    """
    주가 데이터 다운로드 및 캐싱

    Rate limit: 5/minute (설정 가능)
    """
    logger.info(f"Ingest started: max_tickers={max_tickers}, force_refresh={force_refresh}, days={req.days}")

    try:
        # 1) 티커 로드(캐시/리프레시 지원)
        tickers = get_tickers(max_count=max_tickers, force_refresh=force_refresh)
        logger.info(f"Loaded {len(tickers)} tickers")

        # 2) 가격 다운로드(견고 버전 우선)
        try:
            from .data_io import download_ohlc_robust
            raw, ok = download_ohlc_robust(tickers, period="2y")
            if raw is None or raw.empty:
                raise HTTPException(500, "가격 데이터를 가져오지 못했습니다.")
        except ImportError:
            logger.warning("download_ohlc_robust not found, using fallback")
            raw = download_ohlc(tickers, period="2y")
            ok = tickers

        # 3) 기간 슬라이싱 & MA20 계산
        raw = last_n_days(raw, n=req.days)
        ma20 = compute_ma20(raw)
        logger.info(f"MA20 calculated for {len(ma20)} tickers")

        # 4) 디스크 캐시 저장
        p = save_ma20_parquet(ma20)
        save_meta({
            "tickers": list(ma20.keys()),
            "file": p,
            "days": req.days,
            "ts": time.time(),
            "ok_count": len(ok)
        })
        logger.info(f"Data saved to {p}")

        # 5) 메모리 캐시 준비(행렬/티커/정규화 맵)
        matrix, T = dict_to_matrix(ma20, target_len=CACHE["target_len"])
        norm_map = {t: matrix[i, :].tolist() for i, t in enumerate(T)}

        with CACHE_LOCK:
            CACHE.update({"matrix": matrix, "tickers": T, "norm_map": norm_map})

        logger.info(f"Ingest completed: {len(T)} tickers cached")
        return {"tickers_count": len(T), "ok_count": len(ok), "target_len": CACHE["target_len"]}

    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        raise HTTPException(500, f"Ingest 실패: {str(e)}")

@app.post("/refresh_tickers")
def refresh_tickers(max_tickers: int = Query(5000, ge=10, le=5000)):
    """
    NASDAQ API에서 강제로 티커를 다시 받아와 캐시 파일만 갱신합니다. (가격 다운로드는 아님)
    """
    logger.info(f"Refreshing tickers: max_tickers={max_tickers}")
    try:
        syms = get_tickers(max_count=max_tickers, force_refresh=True)
        logger.info(f"Tickers refreshed: {len(syms)} symbols")
        return {"refreshed": len(syms)}
    except Exception as e:
        logger.error(f"Refresh tickers failed: {e}")
        raise HTTPException(500, f"티커 갱신 실패: {str(e)}")

@app.post("/similar", response_model=SimilarResponse)
@limiter.limit(settings.rate_limit_similar)
def similar(request: Request, req: SketchRequest):
    """
    스케치와 유사한 종목 검색

    Rate limit: 20/minute (설정 가능)
    """
    logger.info(f"Similar search started: sketch length={len(req.y)}")

    try:
        # 캐시 없거나 norm_map 미구성 → 디스크에서 불러와 구성
        if CACHE["matrix"] is None or CACHE.get("norm_map") is None:
            logger.info("Cache miss, loading from disk...")
            df = load_ma20_parquet()
            if df is None or df.empty:
                raise HTTPException(400, "먼저 /ingest로 데이터 캐시를 준비하세요.")

            ma20 = {c: df[c].dropna() for c in df.columns}
            matrix, T = dict_to_matrix(ma20, target_len=req.target_len)
            norm_map = {t: matrix[i, :].tolist() for i, t in enumerate(T)}

            with CACHE_LOCK:
                CACHE.update({"matrix": matrix, "tickers": T, "target_len": req.target_len, "norm_map": norm_map})

            logger.info(f"Cache loaded: {len(T)} tickers")

        # 스케치 정규화
        y = np.array(req.y, dtype=float)
        sketch_vec = normalize_pipeline(y, target_len=CACHE["target_len"])
        logger.debug(f"Sketch normalized to {len(sketch_vec)} points")

        # NaN 체크 및 제거
        if np.any(np.isnan(sketch_vec)):
            logger.warning("NaN detected in sketch_vec, cleaning...")
            sketch_vec = np.nan_to_num(sketch_vec, nan=0.0)

        # Top5 랭킹
        pairs = rank_top_k(sketch_vec, CACHE["matrix"], CACHE["tickers"], k=5)
        logger.info(f"Top 5 matches found: {[t for t, _ in pairs]}")

        # 응답(오버레이용 정규화 시리즈 포함)
        items = []
        for i, (t, s) in enumerate(pairs):
            series_norm = CACHE["norm_map"].get(t)
            if series_norm is None:
                idx = CACHE["tickers"].index(t)
                series_norm = CACHE["matrix"][idx, :].tolist()

            # NaN 제거 (리스트인 경우)
            if isinstance(series_norm, list):
                series_norm = [0.0 if (isinstance(x, float) and not np.isfinite(x)) else x for x in series_norm]

            # sketch_norm도 NaN 제거
            sketch_norm_list = [0.0 if not np.isfinite(x) else float(x) for x in sketch_vec]

            # 회사 이름 가져오기
            company_name = CACHE.get("ticker_info", {}).get(t, t)

            items.append(SimilarResponseItem(
                ticker=t,
                name=company_name,
                score=float(s) if np.isfinite(s) else 0.0,
                rank=i+1,
                series_norm=series_norm,       # 해당 종목 MA20 (정규화)
                sketch_norm=sketch_norm_list   # 스케치 (정규화)
            ))

        return SimilarResponse(items=items)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Similar search failed: {e}")
        raise HTTPException(500, f"유사도 검색 실패: {str(e)}")

@app.post("/similar_db", response_model=SimilarResponse)
@limiter.limit(settings.rate_limit_similar)
def similar_db(request: Request, req: SketchRequest):
    """
    DB 기반 스케치 유사도 검색 (PostgreSQL graph_segments 사용)

    Rate limit: 20/minute (설정 가능)
    """
    logger.info(f"Similar DB search started: sketch length={len(req.y)}")

    try:
        # PostgreSQL에서 벡터 세그먼트 로드
        if settings.data_source != "postgresql":
            raise HTTPException(400, "data_source가 'postgresql'로 설정되어야 합니다.")

        # 스케치를 먼저 128차원으로 정규화
        y = np.array(req.y, dtype=float)
        sketch_vec = normalize_pipeline(y, target_len=128)
        logger.debug(f"Sketch normalized to 128 dimensions")

        # NaN 체크 및 제거
        if np.any(np.isnan(sketch_vec)):
            logger.warning("NaN detected in sketch_vec, cleaning...")
            sketch_vec = np.nan_to_num(sketch_vec, nan=0.0)

        # pgvector를 사용하여 DB에서 상위 100개 세그먼트를 가져옴 (1차 필터링)
        vectors, tickers, metadata = db_io.fetch_top_k_segments(sketch_vec, ma_type="MA20", limit=100)

        if len(vectors) == 0:
            raise HTTPException(400, "DB에 저장된 벡터 세그먼트가 없습니다.")

        logger.info(f"Loaded top {len(vectors)} segments from DB using pgvector")

        # 가져온 후보군 100개에 대해서만 정밀 평가(DTW, Pearson 등) 적용하여 Top 5 랭킹
        pairs = rank_top_k(sketch_vec, vectors, tickers, k=5)
        logger.info(f"Top 5 matches found: {[t for t, _ in pairs]}")

        # 응답 구성 (메타데이터 포함)
        items = []
        for i, (t, s) in enumerate(pairs):
            # 해당 티커의 첫 번째 매칭 찾기 (같은 티커의 여러 세그먼트 중)
            ticker_idx = tickers.index(t)
            series_norm = vectors[ticker_idx, :].tolist()
            meta = metadata[ticker_idx]

            # NaN 제거
            if isinstance(series_norm, list):
                series_norm = [0.0 if (isinstance(x, float) and not np.isfinite(x)) else x for x in series_norm]

            sketch_norm_list = [0.0 if not np.isfinite(x) else float(x) for x in sketch_vec]

            # 회사 이름 가져오기
            company_name = CACHE.get("ticker_info", {}).get(t, t)

            items.append(SimilarResponseItem(
                ticker=t,
                name=company_name,
                score=float(s) if np.isfinite(s) else 0.0,
                rank=i+1,
                series_norm=series_norm,       # DB 세그먼트 (128차원)
                sketch_norm=sketch_norm_list   # 스케치 (128차원)
            ))

        return SimilarResponse(items=items)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Similar DB search failed: {e}")
        raise HTTPException(500, f"DB 유사도 검색 실패: {str(e)}")

@app.post("/analyze_pattern")
@limiter.limit("30/minute") # Setting a somewhat high limit for now
def route_analyze_pattern(request: Request, req: SketchRequest):
    """
    AI를 이용해 스케치 패턴 분석 (Alpha)
    """
    try:
        y = np.array(req.y, dtype=float)
        sketch_vec = normalize_pipeline(y, target_len=128)
        if np.any(np.isnan(sketch_vec)):
            sketch_vec = np.nan_to_num(sketch_vec, nan=0.0)
            
        result = analyze_sketch_pattern(sketch_vec.tolist())
        return result
    except Exception as e:
        logger.error(f"Pattern analysis failed: {e}")
        raise HTTPException(500, f"패턴 분석 실패: {str(e)}")

