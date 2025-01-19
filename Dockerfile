FROM python:3.12-slim

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

# 環境変数でデフォルトのワーカー数を指定
ENV WORKERS=1

# コンテナ起動時のコマンド
CMD ["sh", "-c", "gunicorn -w ${WORKERS} -b 0.0.0.0:5000 --timeout 3 app:app"]
