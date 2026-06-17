import asyncio
from httpx import AsyncClient

async def run():
    async with AsyncClient() as client:
        # Assuming the FastAPI server is running on localhost:8080 (from config)
        try:
            resp = await client.post("http://localhost:8080/ingest?max_tickers=100")
            print(resp.json())
        except Exception as e:
            print("Server might not be running locally:", e)

if __name__ == "__main__":
    asyncio.run(run())
