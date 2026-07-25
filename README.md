# Face Detector

[![License: ISC](https://img.shields.io/github/license/nana4rider/face-detector)](LICENSE)
![GitHub Actions Test](https://github.com/nana4rider/ace-detector/actions/workflows/test.yml/badge.svg)
![GitHub Actions Release](https://github.com/nana4rider/ace-detector/actions/workflows/release.yml/badge.svg)

## 概要

画像から顔画像を切り取るAPIです。

コーディングはAPIドキュメントも全てChatGPTに任せており、自分では1行も書かないことを目標にしています🤖

## ドキュメント

- [API Document](https://nana4rider.github.io/openapi-ui/?face-detector)

OpenAPI仕様はFastAPIのコードから生成した次のファイルを公開しています。

- [OpenAPI JSON](docs/openapi.json)

実行中のアプリケーションでは `/openapi.json`、`/docs`、`/redoc` を公開しません。
API定義を変更した場合は、次のコマンドで仕様ファイルを更新してください。

```sh
.venv/bin/python scripts/export_openapi.py
```

VS Codeの「タスクの実行」から `OpenAPI: JSONを生成` を選ぶこともできます。

アプリケーションのバージョンとPython依存関係は `pyproject.toml` で管理します。
リリース時は `[project]` の `version` を更新してください。FastAPIのOpenAPIバージョンと
GitHub Actionsのリリースタグにも同じ値が使われます。

`POST /detect` の `scoreThreshold` は、MediaPipeが返した検出結果のscoreを
絞り込むためのパラメータです。MediaPipeモデル自体の検出閾値は
`FACE_DETECTION_CONFIDENCE` 環境変数で起動時に設定します。
顔検出にはMediaPipe Tasks APIと `models/blaze_face_short_range.tflite` を使用します。
別の互換モデルを使う場合は、`FACE_DETECTION_MODEL` にモデルファイルのパスを
指定してください。

## VS Code + WSLでのローカル実行

Pythonパッケージはプロジェクト直下の `.venv` にだけインストールします。
初回のみ、WSLターミナルで次を実行してください。

```sh
sudo apt update
sudo apt install -y python3-venv libegl1 libgl1 libgles2 libglib2.0-0

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --editable ".[dev]"
.venv/bin/pre-commit install --hook-type pre-commit --hook-type pre-push
```

VS Codeを一度再読み込みすると `.venv/bin/python` が自動選択されます。
「実行とデバッグ」から `FastAPI: 開発サーバー` を選んでF5を押すか、
WSLターミナルで次を実行します。

```sh
source .venv/bin/activate
uvicorn face_detector.main:app --host 127.0.0.1 --port 5000 --reload
```

起動後は <http://localhost:5000/detect> で動作を確認できます。
仮想環境を終了するときは `deactivate`、Pythonパッケージをすべて削除するときは
`.venv` ディレクトリを削除します。

VS Codeの「タスクの実行」から以下を順に実行することもできます。

1. `Python: 仮想環境を作成`
2. `Python: 依存パッケージをインストール`
3. `Git: pre-commitフックを有効化`

コミット時にはpre-commitがRuffのimport整理とフォーマットを自動実行します。
修正が発生した場合はコミットが一度停止するため、差分を確認して再度コミットして
ください。`git push` の前にはpytestが自動実行されます。
リポジトリ全体を手動確認する場合は次を実行できます。

```sh
.venv/bin/pre-commit run --all-files
```

`WORKERS` を増やすと各プロセスがMediaPipeモデルを保持するため、メモリ使用量も
ほぼ比例して増えます。省メモリ運用ではデフォルトの1ワーカーを推奨します。
