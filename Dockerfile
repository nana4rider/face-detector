FROM python:3.14-slim

# 作業ディレクトリの設定
WORKDIR /app

# MediaPipeが依存するOpenCV wheelの最小ランタイムライブラリ
RUN apt-get update \
    && apt-get install -y --no-install-recommends libegl1 libgl1 libgles2 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# プロジェクトと必要なPythonパッケージをインストール
COPY pyproject.toml README.md LICENSE ./
COPY face_detector ./face_detector
RUN pip install --no-cache-dir .

# アプリケーションコードをコピー
COPY . .

# コンテナ起動時のコマンド
CMD ["sh", "-c", "uvicorn face_detector.main:app --host 0.0.0.0 --port 5000 --workers ${WORKERS:-1} --timeout-keep-alive 3"]
