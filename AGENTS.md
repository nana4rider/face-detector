# エージェント向けリポジトリガイド

## プロジェクト概要

このリポジトリは、アップロードされた画像から最もscoreの高い顔を検出し、
JPEGとして返す、省メモリを重視したFastAPIサービスです。ユーザーから明示的に
破壊的変更を依頼されない限り、既存のHTTP APIとの互換性を維持してください。

アプリケーションはPython 3.12以上とMediaPipe Tasks APIを使用します。
本番DockerとCIでは現在Python 3.14を使用しています。

## ソース構成

- `face_detector/main.py`: FastAPIアプリケーション生成、lifespan、例外処理
- `face_detector/api.py`: HTTPルート、フォーム解析、画像読込、切り抜き、レスポンス
- `face_detector/detection.py`: MediaPipeの所有と顔選択・JPEG変換
- `face_detector/config.py`: 環境変数とリポジトリ内アセットのパス
- `face_detector/static/index.html`: `GET /detect` で返すブラウザUI
- `face_detector/models/`: MediaPipeモデルと出典・ライセンス情報
- `scripts/export_openapi.py`: 静的OpenAPIドキュメントの生成
- `tests/`: 実際のMediaPipeモデルを初期化しない高速テスト

モノリシックな `app.py` を再作成しないでください。HTTP、検出処理、設定、
アプリケーション生成の責務分離を維持してください。

## セットアップとコマンド

プロジェクトローカルの仮想環境を使用し、依存パッケージをグローバル環境へ
インストールしないでください。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --editable ".[dev]"
.venv/bin/pre-commit install --hook-type pre-commit --hook-type pre-push
```

Pythonコードを変更したら、引き渡し前にすべて実行してください。

```sh
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/basedpyright
.venv/bin/python -m pytest
```

Ruffによる安全なフォーマットとimport修正は、次のコマンドで適用できます。

```sh
.venv/bin/ruff check --fix .
.venv/bin/ruff format .
```

pre-commitフックはRuffとbasedpyrightを実行します。pre-pushフックは加えて
pytestを実行します。CIも `.github/workflows/test.yml` で同じ検査を行います。

## API互換性

公開ルートは次のとおりです。

- `GET /`（`/detect` へリダイレクト。意図的にOpenAPI対象外）
- `POST /detect`
- `GET /detect`（HTML UI。意図的にOpenAPI対象外）
- `GET /icon.png`（favicon。意図的にOpenAPI対象外）
- `GET /health`

実行中の `/openapi.json`、`/docs`、`/redoc` は無効のままにしてください。
公開するスキーマは `docs/openapi.json` に静的生成します。

`POST /detect` のmultipartフィールド名はAPI契約の一部です。

- `file`
- `scoreThreshold`
- `minSize`
- `startX`
- `startY`
- `endX`
- `endY`

`api.py` のcamelCaseパラメータ名は互換性のため意図的に使用しています。
ユーザーの明示的な承認なしに名前を変えたり、互換エイリアスを追加・削除したり
しないでください。特に旧リクエストパラメータ `confidence` は意図的に廃止されて
いるため、暗黙に復活させないでください。

成功時は `image/jpeg` を返し、次のレスポンスヘッダーを付与します。

- `X-Image-Width`
- `X-Image-Height`

エラーレスポンスの形式と既存の日本語メッセージも外部から観測可能なAPI仕様です。
依頼にAPI変更が含まれない限り維持してください。

ルート、パラメータ、レスポンス、アプリケーションバージョンを変更した場合は、
次を実行してください。

```sh
.venv/bin/python scripts/export_openapi.py
```

`docs/openapi.json` に差分が生じた場合は、内容を確認してコミットしてください。

## 顔検出とパフォーマンスの不変条件

UvicornのワーカーごとにFastAPIのlifespan起動時に1つだけ `FaceDetector` を
生成してください。HTTPリクエストごとにMediaPipe検出器を生成してはいけません。
モデル初期化は高コストなため、意図的にリクエスト処理から分離されています。

MediaPipe検出器へのアクセスは `threading.Lock` で保護されています。使用中の
MediaPipe Tasks APIモードが並行呼び出しに対して安全であるという根拠がない限り、
ロックを削除しないでください。

次の構成は意図的なものです。

- MediaPipe Tasksの `FaceDetector`
- `RunningMode.IMAGE`
- 明示的なCPU delegate
- デフォルトモデル `face_detector/models/blaze_face_short_range.tflite`

次の2つの閾値を混同しないでください。

- `FACE_DETECTION_CONFIDENCE`: 起動時に一度だけ設定し、MediaPipeモデルの
  `min_detection_confidence` に渡す値
- リクエストの `scoreThreshold`: 推論後、MediaPipeが返したcategory scoreに
  適用する値

`scoreThreshold` のためにリクエストごとにモデルを生成してはいけません。

`WORKERS` のデフォルトは1です。ワーカーを増やすたびに別のモデルが読み込まれ、
メモリ使用量が大きく増えます。デフォルトのワーカー数を増やしたり、同一プロセス内に
モデルを重複生成したりしないでください。

NumPyで切り抜く前のbounding box補正を維持してください。`cv2.imdecode()` と
`cv2.imencode()` は失敗しうる処理として扱い、エラー処理を残してください。

## 設定とパッケージ管理

`pyproject.toml` を次の情報の唯一の管理元とします。

- アプリケーションバージョン
- 実行時・開発時の依存関係
- Ruff設定
- pytest設定
- Pyright／basedpyright設定

`requirements.txt` や `version.txt` を再導入しないでください。バージョンは
現在の `main.py` と同様に `importlib.metadata` から取得します。

使用する環境変数は次のとおりです。

- `FACE_DETECTION_MODEL`: 互換モデルへの任意のパス
- `FACE_DETECTION_CONFIDENCE`: 起動時のモデル閾値。デフォルトは `0.5`
- `WORKERS`: Docker内のUvicornプロセス数。デフォルトは `1`

WSL実行にはREADME記載のOSパッケージが必要です。特に `libegl1`、`libgl1`、
`libgles2`、`libglib2.0-0` はPython依存関係では代替できない共有ライブラリです。

Docker構成を変更する場合、`.dockerignore` にあるファイルはDockerfileで明示的に
`COPY` しても利用できないことに注意してください。`README.md` はパッケージの
メタデータ生成時に必要で、`LICENSE` は配布メタデータに含める必要があります。

## テスト方針

pytestを使用し、テストは `tests/` 以下に配置します。振る舞いを変えた場合は、
特にAPI互換性、パラメータ検証、顔の絞り込み、bounding box処理、
レスポンスヘッダーのテストを追加・更新してください。

通常のテストは高速かつ決定的に保ってください。

- 単体テストで実際のMediaPipeモデルを初期化しない
- 偽の検出結果を使用するか `detect_face` を差し替える
- ネットワーク上のfixtureではなく、生成したNumPy画像を使用する
- basedpyright／Pylanceが安全性を証明できるよう、OptionalやUnionを明示的に
  絞り込む

実モデルのスモークテストを追加する場合は、モデルアセットとOS共有ライブラリが
必要になるため、integration testとして通常テストから分離してください。

## Workflowとリリース

`.github/workflows/test.yml` は再利用可能なWorkflowで、`main` 向けのPRでも
実行されます。`.github/workflows/release.yml` は `main` へのpush時に実行され、
タグ作成、Dockerイメージ公開、デプロイより前にtest Workflowの成功を必須とします。

リリースバージョンは `pyproject.toml` の `[project].version` から取得します。
FastAPI／OpenAPIのバージョンとGitタグは同じ値から導出してください。

CIを通すためだけにRuff、basedpyright、pytestやReleaseジョブの依存関係を
弱めたり迂回したりしないでください。根本原因を修正するか、ルール変更の理由を
明記してください。

## 変更時の規律

dirtyなworktreeにある無関係なユーザー変更を保持してください。変更範囲を限定し、
依頼されていない互換性破壊を避けてください。ユーザーから依頼されない限り、
コミットやpushを行わないでください。コミット前にはステージ済み差分を確認し、
インストール済みフックを実行させてください。
