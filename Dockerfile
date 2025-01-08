FROM python:3.13-slim

# 作業ディレクトリの設定
WORKDIR /app

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 必要なPythonパッケージをインストール
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY . .

# コンテナ起動時のコマンド
# CMD ["python", "app.py"]
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
