# 1. 프론트엔드 빌드 스테이지
FROM node:20 AS frontend-builder
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# 2. 백엔드 및 최종 실행 스테이지
FROM python:3.10-slim
WORKDIR /app

# 시스템 라이브러리 설치 (PyCaret 등에 필요할 수 있음)
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# 파이썬 패키지 설치
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 백엔드 코드 복사
COPY backend/ ./backend/

# 프론트엔드 빌드 결과물(dist) 복사
COPY --from=frontend-builder /app/dist ./dist

# 포트 개방 및 실행
EXPOSE 8080
WORKDIR /app/backend
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
