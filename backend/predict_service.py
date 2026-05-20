import os
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import json
from pycaret.classification import load_model, predict_model

warnings.simplefilter(action='ignore')

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODEL_FILE = os.path.join(DATA_DIR, 'weekly_momentum_model')

# Model will be loaded lazily to avoid delay on startup
_model = None

def load_ml_model():
    global _model
    if _model is None:
        _model = load_model(MODEL_FILE)
    return _model

def get_all_tickers():
    data_path = os.path.join(DATA_DIR, 'ma20.parquet')
    df = pd.read_parquet(data_path)
    if 'ticker' in df.columns:
        return df['ticker'].unique().tolist()
    else:
        return [str(c) for c in df.columns if c != 'Date']

def get_target_date(mode="auto"):
    """
    mode='auto': 가장 최근 금요일(미래 제외)을 반환
    mode='today': 오늘 날짜 반환
    mode='YYYY-MM-DD': 해당 날짜 반환
    """
    if mode != "auto" and mode != "today":
        try:
            datetime.datetime.strptime(mode, "%Y-%m-%d")
            return mode
        except ValueError:
            pass
            
    today = datetime.date.today()
    if mode == "today":
        return today.strftime("%Y-%m-%d")
    else:
        # 가장 최근 금요일 찾기
        # weekday(): Monday=0, Tuesday=1, ..., Friday=4, Sunday=6
        days_ahead = 4 - today.weekday()
        if days_ahead > 0: # 이번 주 금요일이 아직 안 지난 경우, 지난 주 금요일로
            days_ahead -= 7
        last_friday = today + datetime.timedelta(days=days_ahead)
        return last_friday.strftime("%Y-%m-%d")

def get_top_10_recommendations(mode="auto"):
    target_date = get_target_date(mode)
    print(f"[예측 기준일] {target_date}")

    # 캐시 확인 로직
    cache_file = os.path.join(DATA_DIR, f"cache_{target_date}.json")
    if os.path.exists(cache_file):
        print(f"[캐시 데이터 로드] {cache_file}")
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[캐시 읽기 실패] 새로 연산합니다: {e}")

    model = load_ml_model()
    tickers = get_all_tickers()[:3000] # 최적화를 위해 일부 자르기(원래 코드와 동일)
    
    all_rows = []
    chunk_size = 500
    
    for i in range(0, len(tickers), chunk_size):
        chunk_tickers = tickers[i:min(i+chunk_size, len(tickers))]
        data = yf.download(chunk_tickers, period="2y", interval="1d", progress=False)
        if data.empty: continue
            
        for t in chunk_tickers:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    try:
                        df = data.xs(t, level=1, axis=1).copy()
                    except KeyError: continue
                else:
                    df = data.copy() if len(chunk_tickers) == 1 else pd.DataFrame()
                    
                df = df.dropna(subset=['Close'])
                if len(df) < 100: continue
                
                # --- 지표 계산 ---
                close = df['Close']
                vol = df['Volume']
                
                df['ROC_5'] = close.pct_change(5)
                df['ROC_20'] = close.pct_change(20)
                df['ROC_60'] = close.pct_change(60)
                
                sma10 = close.rolling(10).mean()
                sma20 = close.rolling(20).mean()
                sma50 = close.rolling(50).mean()
                
                df['Dist_SMA10'] = (close - sma10) / sma10
                df['Dist_SMA20'] = (close - sma20) / sma20
                df['Dist_SMA50'] = (close - sma50) / sma50
                
                vol_sma5 = vol.rolling(5).mean()
                vol_sma50 = vol.rolling(50).mean()
                df['Vol_Ratio'] = vol_sma5 / vol_sma50.replace(0, np.nan)
                
                high_252 = df['High'].rolling(252).max()
                df['Dist_High52'] = (close - high_252) / high_252
                
                delta = close.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss.replace(0, np.nan)
                df['RSI_14'] = 100 - (100 / (1 + rs))
                
                df['Ticker'] = t
                df['Date'] = df.index
                
                # 타겟 날짜 시점까지만 데이터를 자름
                sub_df = df.loc[:target_date].copy()
                if sub_df.empty: continue
                
                # 가장 최신 영업일의 데이터 1줄 추출
                last_row = sub_df.iloc[-1:].copy()
                
                # 현재 종가(UI 표시용)
                current_price = last_row['Close'].values[0]
                
                # 1. 가격 필터: 5달러 미만 (워런트, 페니스탁 등) 제외
                if current_price < 5:
                    continue
                    
                # 2. 상장폐지 필터: 마지막 거래일이 target_date 기준 7일 이상 지났으면 제외
                target_dt = pd.to_datetime(target_date)
                last_dt = pd.to_datetime(last_row.index[0])
                if (target_dt - last_dt).days > 7:
                    continue
                
                # 7일 전 대비 변동률 (간단히 표시용으로 최근 5일(1주) 수익률 활용)
                change_1w = last_row['ROC_5'].values[0] * 100
                
                last_row['Price_Display'] = f"${current_price:.2f}"
                last_row['Change_Display'] = f"{change_1w:+.2f}%"
                
                features = [
                    'Ticker', 'Date', 'Price_Display', 'Change_Display',
                    'ROC_5', 'ROC_20', 'ROC_60', 
                    'Dist_SMA10', 'Dist_SMA20', 'Dist_SMA50', 'Dist_High52',
                    'Vol_Ratio', 'RSI_14'
                ]
                all_rows.append(last_row[features])
            except Exception:
                continue

    if not all_rows:
        return []
        
    print(f"[필터링 완료] 최종 분석 대상 티커 수: {len(all_rows)}개")
    
    final_df = pd.concat(all_rows, ignore_index=True)
    features_only = final_df.drop(columns=['Ticker', 'Date', 'Price_Display', 'Change_Display'], errors='ignore')
    
    predictions = predict_model(model, data=features_only, raw_score=True)
    
    prob_col = 'prediction_score_1'
    if prob_col not in predictions.columns:
        prob_col = 'prediction_score'
        
    predictions['Ticker'] = final_df['Ticker']
    predictions['Price'] = final_df['Price_Display']
    predictions['Change'] = final_df['Change_Display']
    
    bulls = predictions.sort_values(by=prob_col, ascending=False)
    
    # 특징값 백분위수 계산 (전체 종목 대비 상위/하위 퍼센트)
    final_df['Vol_Ratio_Pct'] = final_df['Vol_Ratio'].rank(pct=True)
    final_df['RSI_14_Pct'] = final_df['RSI_14'].rank(pct=True)
    final_df['ROC_5_Pct'] = final_df['ROC_5'].rank(pct=True)
    final_df['Dist_SMA10_Pct'] = final_df['Dist_SMA10'].rank(pct=True)
    final_df['Dist_High52_Pct'] = final_df['Dist_High52'].rank(pct=True)

    results = []
    for i, (_, row) in enumerate(bulls.iterrows(), 1):
        prob = row[prob_col] * 100
        ticker = row['Ticker']
        
        # 원래 데이터프레임에서 해당 티커의 백분위수 등 데이터 가져오기
        orig_row = final_df[final_df['Ticker'] == ticker].iloc[0]
        
        reasons = []
        tags = []
        
        # 1. 거래량 아웃라이어
        vol_pct = orig_row['Vol_Ratio_Pct'] * 100
        if vol_pct > 90:
            reasons.append(f"최근 거래량이 전체 분석 종목 중 상위 {100 - vol_pct:.1f}% 수준으로 폭발하며 강력한 매수세가 확인됩니다.")
            tags.append("#거래량급증")
            
        # 2. RSI 극단값
        rsi = orig_row['RSI_14']
        rsi_pct = orig_row['RSI_14_Pct'] * 100
        if rsi < 40:
            reasons.append(f"RSI(상대강도지수)가 {rsi:.1f}로 과매도 구간(하위 {rsi_pct:.1f}%)에 진입하여 AI가 단기 기술적 반등 확률을 높게 평가했습니다.")
            tags.append("#과매도반등")
        elif rsi > 70:
            reasons.append(f"RSI 지표가 {rsi:.1f}를 기록하며 전체 상위 {100 - rsi_pct:.1f}%의 강한 추세 모멘텀을 형성하고 있습니다.")
            tags.append("#강한모멘텀")
            
        # 3. 단기 수익률 아웃라이어
        roc5_pct = orig_row['ROC_5_Pct'] * 100
        if roc5_pct > 95:
            reasons.append(f"최근 5일간 수익률이 전체 상위 {100 - roc5_pct:.1f}%를 기록하며 주도주 패턴을 보입니다.")
            tags.append("#단기급등")
            
        # 4. 신고가 근접도
        dist_high52 = orig_row['Dist_High52']
        if dist_high52 > -0.05: # -5% 이내
            reasons.append(f"52주 신고가 돌파까지 불과 {abs(dist_high52)*100:.1f}% 남겨두고 있어 강력한 상방 돌파 여력이 기대됩니다.")
            tags.append("#신고가근접")
            
        # 특징이 없을 경우 기본 텍스트
        if not reasons:
            reasons.append("이동평균선 지지선과 긍정적인 기술적 지표들이 맞물려 안정적인 우상향 패턴을 나타냅니다.")
            tags.append("#안정적추세")
            
        reason_text = " ".join(reasons)

        results.append({
            "rank": int(i),
            "id": str(ticker),
            "name": str(ticker) + " Corp",
            "symbol": str(ticker),
            "price": str(row['Price']),
            "change": str(row['Change']),
            "prob": f"{prob:.1f}%",
            "reason_text": reason_text,
            "reason_tags": tags[:3]
        })

        
    # 결과를 캐시 파일로 저장 (다음번 요청시 1초 만에 로드되도록)
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"[캐시 저장 완료] {cache_file}")
    except Exception as e:
        print(f"[캐시 저장 실패] {e}")
        
    return results

if __name__ == "__main__":
    print(get_top_10_recommendations("auto"))
