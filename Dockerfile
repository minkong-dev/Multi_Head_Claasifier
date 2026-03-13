FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# 작업 디렉토리
WORKDIR /app

# 시스템 패키지 
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 전체 코드 복사 
COPY . .

# 포트 노출
EXPOSE 8000

# 실행 명령 
CMD ["python", "backend/api/main.py"]