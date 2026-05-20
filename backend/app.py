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
    # /assets 경로로 js, css 파일 등 서빙
    app.mount("/assets", StaticFiles(directory=os.path.join(dist_dir, "assets")), name="assets")
    
    # 루트 경로(/) 접속 시 index.html 반환
    @app.get("/")
    def read_index():
        return FileResponse(os.path.join(dist_dir, "index.html"))
    
    # /search 경로나 /recommend 경로 등이 있다면 각각 매핑
    @app.get("/search.html")
    def read_search():
        return FileResponse(os.path.join(dist_dir, "search.html"))
        
    @app.get("/recommend.html")
    def read_recommend():
        return FileResponse(os.path.join(dist_dir, "recommend.html"))