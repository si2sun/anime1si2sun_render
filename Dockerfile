# 使用官方 Python 映像作為基礎
FROM python:3.9-slim-buster

# 設定工作目錄
WORKDIR /app

# 將 requirements.txt 複製到工作目錄
COPY requirements.txt .

# 安裝 Python 依賴
RUN pip install --no-cache-dir -r requirements.txt

# 將所有應用程式程式碼複製到工作目錄
COPY . .

# 公開應用程式將運行的埠 (FastAPI 預設使用 8000，但 Render 會提供 $PORT)
EXPOSE 8000

# 啟動應用程式的命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]