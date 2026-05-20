import sys
sys.stdout.reconfigure(encoding='utf-8')
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from predict_service import get_top_10_recommendations
import uvicorn
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="DRS Recommend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/recommend")
def recommend(date: str = Query("auto", description="Date mode: 'auto' (last friday) or 'today'")):
    try:
        results = get_top_10_recommendations(mode=date)
        return {"status": "success", "data": results}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

@app.get("/debug")
def debug():
    import predict_service
    return {
        "app_file": __file__,
        "app_cwd": os.getcwd(),
        "predict_file": predict_service.__file__,
        "predict_code": open(predict_service.__file__, 'r', encoding='utf-8').read()[:200]
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)

# 상위 폴더의 dist 디렉토리를 가리킴 (Vite 빌드 결과물)
dist_dir = os.path.join(os.path.dirname(__file__), "..", "dist")
if os.path.isdir(dist_dir):
    # Search API 마운트 및 warmup 트리거 추가
    from search_api.main import app as search_app, warmup as search_warmup
    
    @app.on_event("startup")
    def startup_event():
        search_warmup()
        
    app.mount("/api/search", search_app)

    # 나머지 모든 정적 파일(HTML, JS, CSS, MP3, 3D 모델 등) 서빙
    app.mount("/", StaticFiles(directory=dist_dir, html=True), name="static")