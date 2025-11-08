FROM python:3.13-slim

# 作業ディレクトリの設定
WORKDIR /app

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    ffmpeg \
    libsm6 \
    libxext6 && apt-get clean

# 必要なPythonパッケージをインストール
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY . .

# コンテナ起動時のコマンド
CMD ["sh", "-c", "gunicorn -w ${WORKERS:-1} -b 0.0.0.0:5000 --timeout 3 app:app"]
