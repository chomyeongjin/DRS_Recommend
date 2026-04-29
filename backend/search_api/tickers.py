# app/tickers.py
import requests, time, json, re
import logging
from pathlib import Path
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
CACHE_FILE = DATA_DIR / "tickers_nasdaq.json"
TICKER_INFO_CACHE = DATA_DIR / "ticker_info.json"

# Use config if available, otherwise fallback
try:
    from .config import settings
    CACHE_TTL_SEC = settings.cache_ttl_sec
except ImportError:
    CACHE_TTL_SEC = 24 * 3600  # 캐시 유효기간 1일

HEADERS = {"User-Agent": "Mozilla/5.0"}
NASDAQ_API = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=9999"

_symbol_ok = re.compile(r"^[A-Z][A-Z0-9\.\-]*$")

def _normalize_for_yfinance(symbol: str) -> str:
    s = symbol.strip().upper().replace(".", "-")
    return s

def _looks_like_equity(symbol: str) -> bool:
    if not _symbol_ok.match(symbol):
        return False
    bad_prefix = ("^", "$")
    if symbol.startswith(bad_prefix):
        return False
    s = _normalize_for_yfinance(symbol)
    bad_fragments = ("-WT", "-WS", "-W", "-R", "-U")  # SPAC/워런트 등 제외
    if any(s.endswith(f) for f in bad_fragments):
        return False
    return True

def fetch_ticker_info_from_nasdaq() -> Dict[str, str]:
    """
    NASDAQ API에서 티커 정보 (심볼 -> 회사명 매핑) 가져오기

    Returns:
        {ticker: company_name} 딕셔너리

    Raises:
        requests.RequestException: API 호출 실패 시
        ValueError: 응답 파싱 실패 시
    """
    try:
        logger.info(f"Fetching ticker info from NASDAQ API: {NASDAQ_API}")
        r = requests.get(NASDAQ_API, headers=HEADERS, timeout=30)
        r.raise_for_status()

        data = r.json()
        if "data" not in data or "table" not in data["data"] or "rows" not in data["data"]["table"]:
            raise ValueError("Invalid response format from NASDAQ API")

        rows = data["data"]["table"]["rows"]
        ticker_info = {}

        for row in rows:
            symbol = row.get("symbol")
            name = row.get("name", symbol)  # Fallback to symbol if no name

            if not symbol or not _looks_like_equity(symbol):
                continue

            # Normalize symbol for yfinance
            normalized = _normalize_for_yfinance(symbol)
            ticker_info[normalized] = name

        logger.info(f"Fetched info for {len(ticker_info)} tickers")
        return ticker_info

    except requests.RequestException as e:
        logger.error(f"Network error while fetching tickers: {e}")
        raise
    except (KeyError, ValueError) as e:
        logger.error(f"Failed to parse NASDAQ API response: {e}")
        raise ValueError(f"NASDAQ API 응답 파싱 실패: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in fetch_ticker_info_from_nasdaq: {e}")
        raise

def fetch_tickers_from_nasdaq() -> List[str]:
    """
    NASDAQ API에서 티커 리스트 가져오기 (하위 호환성 유지)

    Returns:
        정규화된 티커 심볼 리스트
    """
    ticker_info = fetch_ticker_info_from_nasdaq()
    return sorted(ticker_info.keys())

def load_cached_tickers() -> Optional[List[str]]:
    """캐시 파일에서 티커 로드 (TTL 확인)"""
    if CACHE_FILE.exists():
        mtime = CACHE_FILE.stat().st_mtime
        age = time.time() - mtime
        if age < CACHE_TTL_SEC:
            try:
                tickers = json.loads(CACHE_FILE.read_text())
                logger.info(f"Loaded {len(tickers)} tickers from cache (age: {age/3600:.1f}h)")
                return tickers
            except Exception as e:
                logger.warning(f"Failed to load cache file: {e}")
                return None
        else:
            logger.info(f"Cache expired (age: {age/3600:.1f}h > {CACHE_TTL_SEC/3600:.1f}h)")
    return None

def save_cached_tickers(symbols: List[str]) -> None:
    """티커 리스트를 캐시 파일에 저장"""
    try:
        CACHE_FILE.write_text(json.dumps(symbols, indent=2))
        logger.info(f"Saved {len(symbols)} tickers to cache: {CACHE_FILE}")
    except Exception as e:
        logger.error(f"Failed to save cache file: {e}")

def load_ticker_info() -> Optional[Dict[str, str]]:
    """캐시에서 티커 정보 (심볼 -> 회사명) 로드"""
    if TICKER_INFO_CACHE.exists():
        mtime = TICKER_INFO_CACHE.stat().st_mtime
        age = time.time() - mtime
        if age < CACHE_TTL_SEC:
            try:
                info = json.loads(TICKER_INFO_CACHE.read_text())
                logger.info(f"Loaded ticker info for {len(info)} tickers from cache")
                return info
            except Exception as e:
                logger.warning(f"Failed to load ticker info cache: {e}")
    return None

def save_ticker_info(ticker_info: Dict[str, str]) -> None:
    """티커 정보를 캐시 파일에 저장"""
    try:
        TICKER_INFO_CACHE.write_text(json.dumps(ticker_info, indent=2, ensure_ascii=False))
        logger.info(f"Saved ticker info for {len(ticker_info)} tickers to cache")
    except Exception as e:
        logger.error(f"Failed to save ticker info cache: {e}")

def get_ticker_info(force_refresh: bool = False) -> Dict[str, str]:
    """
    티커 정보 (심볼 -> 회사명 매핑) 가져오기

    Args:
        force_refresh: True면 캐시 무시하고 API 호출

    Returns:
        {ticker: company_name} 딕셔너리
    """
    if not force_refresh:
        cached = load_ticker_info()
        if cached:
            return cached

    logger.info(f"Fetching fresh ticker info from NASDAQ (force_refresh={force_refresh})")
    info = fetch_ticker_info_from_nasdaq()
    save_ticker_info(info)
    return info

def get_tickers(max_count: int = 5000, force_refresh: bool = False) -> List[str]:
    """
    NASDAQ 전체 티커 중 상위 max_count개 반환

    Args:
        max_count: 반환할 최대 티커 개수
        force_refresh: True면 캐시 무시하고 API 호출

    Returns:
        티커 심볼 리스트
    """
    if not force_refresh:
        cached = load_cached_tickers()
        if cached:
            logger.info(f"Using cached tickers (requested: {max_count}, available: {len(cached)})")
            return cached[:max_count]

    logger.info(f"Fetching fresh tickers from NASDAQ (force_refresh={force_refresh})")
    syms = fetch_tickers_from_nasdaq()
    save_cached_tickers(syms)
    return syms[:max_count]
