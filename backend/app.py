from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from predict_service import get_top_10_recommendations
import uvicorn

app = FastAPI(title="DRS Recommend API")

# Configure CORS so the Vite frontend (usually on port 5173) can communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/recommend")
def recommend(date: str = Query("auto", description="Date mode: 'auto' (last friday) or 'today'")):
    # Call the ML pipeline
    try:
        results = get_top_10_recommendations(mode=date)
        return {"status": "success", "data": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
