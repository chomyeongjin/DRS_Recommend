import yfinance as yf
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Optional
import time, json

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Constants
MIN_MA_POINTS = 25  # MA20 계산에 필요한 최소 데이터 포인트 (20 + 5 여유)
MA_WINDOW = 20      # 이동평균 윈도우 크기

def download_ohlc(tickers: List[str], period: str = "2y") -> pd.DataFrame:
    """
    yfinance를 통해 OHLC 데이터 다운로드

    Args:
        tickers: 티커 심볼 리스트
        period: 데이터 기간 (예: "1y", "2y", "5y")

    Returns:
        MultiIndex DataFrame (ticker, field)
    """
    logger.info(f"Downloading OHLC data for {len(tickers)} tickers, period={period}")
    df = yf.download(
        tickers, period=period, interval="1d",
        auto_adjust=True, group_by="ticker", threads=True, progress=False
    )
    logger.info(f"Downloaded OHLC data: shape={df.shape}")
    return df

def last_n_days(df: pd.DataFrame, n: int = 365) -> pd.DataFrame:
    """
    최근 N일 데이터만 슬라이싱

    Args:
        df: OHLC DataFrame
        n: 일수

    Returns:
        슬라이싱된 DataFrame
    """
    result = df.loc[df.index >= (df.index.max() - pd.Timedelta(days=n))]
    logger.debug(f"Sliced to last {n} days: {len(result)} rows")
    return result

def compute_ma20(ohlc_multi: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    각 티커별 Close 가격의 20일 이동평균 계산

    Args:
        ohlc_multi: MultiIndex 컬럼 DataFrame (ticker, field)

    Returns:
        {ticker: MA20 Series} 딕셔너리
    """
    out = {}
    # 멀티컬럼: (Ticker, Field)
    for t in ohlc_multi.columns.levels[0]:
        if (t, 'Close') in ohlc_multi.columns:
            s = ohlc_multi[(t, 'Close')].dropna()
            if len(s) >= MIN_MA_POINTS:
                out[t] = s.rolling(MA_WINDOW).mean().dropna()
            else:
                logger.warning(f"Ticker {t} has insufficient data ({len(s)} < {MIN_MA_POINTS}), skipping")

    logger.info(f"MA20 calculated for {len(out)} tickers")
    return out

def save_ma20_parquet(ma_dict: Dict[str, pd.Series]) -> str:
    """
    MA20 데이터를 Parquet 파일로 저장

    Args:
        ma_dict: {ticker: MA20 Series} 딕셔너리

    Returns:
        저장된 파일 경로
    """
    df = pd.DataFrame({t: s for t, s in ma_dict.items()}).dropna(how="all")
    p = DATA_DIR / "ma20.parquet"
    df.to_parquet(p)
    logger.info(f"Saved MA20 data to {p}: {df.shape}")
    return str(p)

def save_meta(meta: dict) -> None:
    """메타데이터를 JSON 파일로 저장"""
    path = DATA_DIR / "meta.json"
    path.write_text(json.dumps(meta, indent=2))
    logger.debug(f"Saved metadata to {path}")

def load_ma20_parquet() -> Optional[pd.DataFrame]:
    """
    Parquet 파일에서 MA20 데이터 로드

    Returns:
        DataFrame 또는 None (파일 없으면)
    """
    p = DATA_DIR / "ma20.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        logger.debug(f"Loaded MA20 data from {p}: {df.shape}")
        return df
    else:
        logger.debug(f"No parquet file found at {p}")
        return None
