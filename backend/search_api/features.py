import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, List

logger = logging.getLogger(__name__)

# Constants
MIN_DATA_POINTS = 30  # dict_to_matrix에서 필터링할 최소 데이터 포인트
ZSCORE_EPSILON = 1e-8  # Z-score 계산 시 division by zero 방지

def resample_series(y: np.ndarray, target_len: int) -> np.ndarray:
    """
    시계열을 고정 길이로 리샘플링 (선형 보간)

    Args:
        y: 입력 시계열 (임의 길이)
        target_len: 목표 길이

    Returns:
        리샘플링된 시계열 (target_len 길이)
    """
    x_old = np.linspace(0, 1, num=len(y))
    x_new = np.linspace(0, 1, num=target_len)
    return np.interp(x_new, x_old, y)

def zscore(y: np.ndarray, eps: float = ZSCORE_EPSILON) -> np.ndarray:
    """
    Z-score 정규화 (평균=0, 표준편차=1) - NaN 안전

    Args:
        y: 입력 시계열
        eps: 표준편차가 0일 때 사용할 최소값

    Returns:
        정규화된 시계열 (NaN 제거됨)
    """
    # NaN 제거
    if np.any(np.isnan(y)):
        logger.warning(f"NaN detected in zscore input, replacing with mean")
        # NaN을 평균으로 대체
        mask = np.isnan(y)
        y = y.copy()  # 원본 보존
        if np.all(mask):  # 모든 값이 NaN
            logger.error("All values are NaN in zscore input")
            return np.zeros_like(y)
        y[mask] = np.nanmean(y)

    mu, std = y.mean(), y.std()

    # 표준편차가 0이거나 매우 작은 경우 (모든 값이 같음)
    if std < eps:
        logger.debug(f"Low variance detected (std={std}), returning zero-centered array")
        return y - mu  # 평균만 빼고 스케일링 안 함

    result = (y - mu) / std

    # 최종 NaN 체크
    if np.any(np.isnan(result)):
        logger.error("NaN in zscore output, replacing with zeros")
        result = np.nan_to_num(result, nan=0.0)

    return result

def normalize_pipeline(y: np.ndarray, target_len: int) -> np.ndarray:
    """
    정규화 파이프라인: 리샘플링 → Z-score 정규화

    Args:
        y: 입력 시계열
        target_len: 목표 길이

    Returns:
        정규화된 시계열
    """
    y = resample_series(y, target_len)
    y = zscore(y)
    return y

def dict_to_matrix(ma_dict: Dict[str, pd.Series], target_len: int) -> Tuple[np.ndarray, List[str]]:
    """
    MA20 딕셔너리를 정규화된 NumPy 매트릭스로 변환

    Args:
        ma_dict: {ticker: MA20 Series} 딕셔너리
        target_len: 리샘플링 목표 길이

    Returns:
        (정규화 매트릭스, 티커 리스트) 튜플
        - 매트릭스: (N tickers × target_len) shape
        - 티커 리스트: 각 행에 대응하는 티커 심볼
    """
    rows, tickers = [], []
    filtered_count = 0

    for t, s in ma_dict.items():
        y = s.values.astype(float)
        if len(y) < MIN_DATA_POINTS:
            filtered_count += 1
            logger.debug(f"Ticker {t} filtered out: {len(y)} < {MIN_DATA_POINTS} points")
            continue
        rows.append(normalize_pipeline(y, target_len))
        tickers.append(t)

    if filtered_count > 0:
        logger.info(f"Filtered out {filtered_count} tickers with insufficient data")

    matrix = np.vstack(rows)
    logger.info(f"Created matrix: {matrix.shape} ({len(tickers)} tickers × {target_len} points)")
    return matrix, tickers
