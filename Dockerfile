FROM python:3.12-slim

# 作業ディレクトリの設定
WORKDIR /app

# MediaPipeが依存するOpenCV wheelの最小ランタイムライブラリ
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 必要なPythonパッケージをインストール
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY . .

# コンテナ起動時のコマンド
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 5000 --workers ${WORKERS:-1} --timeout-keep-alive 3"]
