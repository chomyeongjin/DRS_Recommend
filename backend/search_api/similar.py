import numpy as np
import logging
from fastdtw import fastdtw
from scipy.stats import pearsonr
from numpy.linalg import norm
from typing import Tuple, List

logger = logging.getLogger(__name__)

def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Dynamic Time Warping 거리 계산

    Args:
        a, b: 비교할 시계열

    Returns:
        DTW 거리 (항상 유효한 float)
    """
    try:
        d, _ = fastdtw(a, b)
        result = float(d)
        return result if np.isfinite(result) else 0.0
    except Exception as e:
        logger.warning(f"DTW calculation failed: {e}")
        return 0.0

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    코사인 유사도 계산 (NaN 안전)

    Args:
        a, b: 비교할 벡터

    Returns:
        코사인 유사도 [-1, 1] 또는 0.0 (계산 실패 시)
    """
    try:
        # NaN 체크
        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            logger.warning("NaN detected in cosine_sim input")
            return 0.0

        norm_a = norm(a)
        norm_b = norm(b)

        # Zero vector 체크
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0

        result = float(np.dot(a, b) / (norm_a * norm_b))
        return result if np.isfinite(result) else 0.0
    except Exception as e:
        logger.warning(f"Cosine similarity calculation failed: {e}")
        return 0.0

def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """
    Pearson 상관계수 계산 (NaN 안전)

    Args:
        a, b: 비교할 시계열

    Returns:
        Pearson 상관계수 [-1, 1] 또는 0.0 (계산 실패 시)
    """
    try:
        # NaN 체크
        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            logger.warning("NaN detected in pearson input")
            return 0.0

        # 표준편차 체크 (pearsonr은 std=0일 때 NaN 반환)
        if np.std(a) < 1e-10 or np.std(b) < 1e-10:
            logger.debug("Zero variance in pearson input")
            return 0.0

        r, _ = pearsonr(a, b)

        # NaN 또는 Inf 체크
        if not np.isfinite(r):
            logger.warning(f"Pearson returned non-finite value: {r}")
            return 0.0

        return float(r)
    except Exception as e:
        logger.warning(f"Pearson calculation failed: {e}")
        return 0.0

def ensemble_score(sketch: np.ndarray, series: np.ndarray,
                   alpha: float = 0.7, beta: float = 0.2, gamma: float = 0.1) -> float:
    """
    앙상블 유사도 스코어 계산 (NaN 안전)

    Args:
        sketch: 사용자 스케치 벡터
        series: 비교 대상 시계열
        alpha: DTW 가중치 (기본 0.7)
        beta: Pearson 가중치 (기본 0.2)
        gamma: Cosine 가중치 (기본 0.1)

    Returns:
        앙상블 스코어 (항상 유효한 float, 범위: 0~1)
    """
    try:
        # 각 메트릭 계산 (모두 NaN-safe)
        d = dtw_distance(sketch, series)

        # DTW 거리 → 유사도 변환 (0~1 범위)
        # 방법: 1 / (1 + normalized_distance)
        # normalized_distance = d / len(sketch)
        d_normalized = d / len(sketch)
        dtw_similarity = 1.0 / (1.0 + d_normalized)

        # Pearson과 Cosine은 -1~1 범위 → 0~1로 변환
        c = pearson(sketch, series)
        c_normalized = (c + 1.0) / 2.0  # -1~1 → 0~1

        s = cosine_sim(sketch, series)
        s_normalized = (s + 1.0) / 2.0  # -1~1 → 0~1

        # 앙상블 스코어 (모두 0~1 범위)
        score = alpha * dtw_similarity + beta * c_normalized + gamma * s_normalized

        # 최종 NaN 체크
        if not np.isfinite(score):
            logger.warning(f"Non-finite ensemble score: {score}, using 0.0")
            return 0.0

        # 0~1 범위 보장
        score = max(0.0, min(1.0, score))

        return float(score)
    except Exception as e:
        logger.error(f"Ensemble score calculation failed: {e}")
        return 0.0

def rank_top_k(sketch_vec: np.ndarray, db_matrix: np.ndarray,
               tickers: List[str], k: int = 5) -> List[Tuple[str, float]]:
    """
    Top-K 유사 종목 랭킹 (NaN 안전)

    Args:
        sketch_vec: 스케치 벡터
        db_matrix: 티커 매트릭스 (N × target_len)
        tickers: 티커 심볼 리스트
        k: 반환할 상위 개수

    Returns:
        [(ticker, score), ...] 리스트 (스코어 내림차순)
    """
    logger.info(f"Ranking top {k} from {len(tickers)} tickers")

    # 모든 스코어 계산
    scores = []
    valid_count = 0

    for i, row in enumerate(db_matrix):
        score = ensemble_score(sketch_vec, row)
        scores.append(score)

        if np.isfinite(score):
            valid_count += 1

    logger.info(f"Valid scores: {valid_count}/{len(scores)}")

    # NaN을 -inf로 대체하여 정렬 시 뒤로 밀림
    scores_array = np.array(scores)
    scores_array = np.nan_to_num(scores_array, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)

    # Top-K 추출
    idx = np.argsort(scores_array)[::-1][:k]

    results = []
    for i in idx:
        score = float(scores_array[i])
        # -inf는 제외
        if score > -np.inf:
            results.append((tickers[i], score))

    logger.info(f"Top {len(results)} results: {[t for t, _ in results]}")
    return results
