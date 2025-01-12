import cv2
import io
from flask import Flask, request, send_file, jsonify, render_template_string
import numpy as np
import mediapipe as mp

app = Flask(__name__)

def detect_face(image_data, params):
    # Mediapipeの顔検出を初期化
    mp_face_detection = mp.solutions.face_detection
    min_confidence = float(params.get("confidence") or 0.5)
    min_size = int(params.get("minSize") or 0)  # 最小サイズ（ピクセル単位）
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=min_confidence)

    # 入力画像をデコード
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return {"error": "入力画像を読み込めませんでした。", "status": 400}

    # Mediapipeで顔検出
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.detections:
        return {"error": "顔が検出されませんでした。", "status": 404}

    # 検出された顔の座標とスコアを取得
    faces = []
    ih, iw, _ = image.shape
    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        score = detection.score[0]  # 信頼度
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
        # 最小サイズ以上の顔のみ追加
        if w >= min_size and h >= min_size:
            faces.append({"bbox": (x, y, w, h), "score": score})

    # フィルタリング後に顔がない場合
    if not faces:
        return {"error": "指定されたサイズ以上の顔が検出されませんでした。", "status": 404}

    # 信頼度順に顔をソート
    faces.sort(key=lambda f: f["score"], reverse=True)

    # 各顔を切り抜き
    face_images = []
    for face in faces:
        x, y, w, h = face["bbox"]
        face_image = image[y:y+h, x:x+w]
        face_images.append(face_image)

    # 横並びに結合
    try:
        combined_image = cv2.hconcat(face_images)
    except cv2.error as e:
        return {"error": f"顔の結合に失敗しました: {str(e)}", "status": 500}

    # バイナリデータを返却
    _, buffer = cv2.imencode('.jpg', combined_image)
    return buffer, combined_image.shape[:2]  # 高さと幅を返却

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "ファイルがアップロードされていません。"}), 400

        file = request.files['file']
        image_data = file.read()

        # 入力画像をデコード
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "入力画像を読み込めませんでした。"}), 400

        # 画像のサイズを取得
        ih, iw, _ = image.shape

        # リクエストパラメータを取得し、空文字列をデフォルト値に置き換え
        params = {
            "confidence": request.form.get("confidence"),  # 信頼度の閾値
            "minSize": request.form.get("minSize"),        # 最小サイズ
            "startX": int(request.form.get("startX") or 0),
            "startY": int(request.form.get("startY") or 0),
            "endX": int(request.form.get("endX") or iw),
            "endY": int(request.form.get("endY") or ih),
        }

        # トリミング処理が必要か確認
        if all(param != "" for param in [request.form.get("startX"), request.form.get("startY"), request.form.get("endX"), request.form.get("endY")]):
            try:
                # パラメータを補正
                start_x = max(0, min(params["startX"], iw))
                start_y = max(0, min(params["startY"], ih))
                end_x = max(start_x, min(params["endX"], iw))
                end_y = max(start_y, min(params["endY"], ih))

                # 画像をトリミング
                image = image[start_y:end_y, start_x:end_x]
            except Exception as e:
                return jsonify({"error": f"切り抜き範囲が無効です: {str(e)}"}), 400

        # 顔検出処理
        result = detect_face(cv2.imencode('.jpg', image)[1].tobytes(), params)
        if isinstance(result, tuple):
            buffer, dimensions = result
            response = send_file(io.BytesIO(buffer), mimetype='image/jpeg')
            response.headers['X-Image-Width'] = dimensions[0]
            response.headers['X-Image-Height'] = dimensions[1]
            return response
        elif isinstance(result, dict):
            return jsonify({"error": result["error"]}), result["status"]
        else:
            return jsonify({"error": "不明なエラーが発生しました。"}), 500

    except Exception as e:
        print(f"エラー: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/detect', methods=['GET'])
def detect_form():
    form_html = '''
    <!doctype html>
    <html lang="ja">
      <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Face Detector</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH" crossorigin="anonymous">
        <script>
          async function handleSubmit(event) {
            event.preventDefault();

            const form = event.target;
            const formData = new FormData(form);

            try {
              const response = await fetch(form.action, {
                method: form.method,
                body: formData
              });

              const resultDiv = document.getElementById('result');
              if (response.ok) {
                const blob = await response.blob();
                const imgUrl = URL.createObjectURL(blob);
                const width = response.headers.get('X-Image-Width');
                const height = response.headers.get('X-Image-Height');
                resultDiv.innerHTML = `<img class="img-fluid" src="${imgUrl}" alt="Detected Face"><p>Width: ${width}px, Height: ${height}px</p>`;
              } else {
                const error = await response.json();
                resultDiv.innerHTML = `<div class="alert alert-danger" role="alert">エラー (${response.status}): ${error.error}</div>`;
              }
            } catch (err) {
              document.getElementById('result').innerHTML = `<div class="alert alert-danger" role="alert">エラー: ${err.message}</div>`;
            }
          }
        </script>
      </head>
      <body class="bg-light">
        <div class="container py-5">
          <h1 class="mb-4">Face Detector</h1>
          <form action="/detect" method="post" enctype="multipart/form-data" onsubmit="handleSubmit(event)" class="card p-4 shadow-sm">
            <div class="mb-3">
              <label for="file" class="form-label">画像ファイル:</label>
              <input type="file" name="file" id="file" accept="image/*" class="form-control" required>
            </div>
            <div class="mb-3">
              <label for="confidence" class="form-label">信頼度閾値 (0.0 - 1.0):</label>
              <input type="number" name="confidence" id="confidence" step="0.1" min="0" max="1" value="0.5" class="form-control">
            </div>
            <div class="mb-3">
              <label for="minSize" class="form-label">最小サイズ (ピクセル):</label>
              <input type="number" name="minSize" id="minSize" value="0" class="form-control">
            </div>
            <div class="row mb-3">
              <div class="col">
                <label for="startX" class="form-label">切り抜き開始 X (省略可能):</label>
                <input type="number" name="startX" id="startX" class="form-control">
              </div>
              <div class="col">
                <label for="startY" class="form-label">切り抜き開始 Y (省略可能):</label>
                <input type="number" name="startY" id="startY" class="form-control">
              </div>
            </div>
            <div class="row mb-3">
              <div class="col">
                <label for="endX" class="form-label">切り抜き終了 X (省略可能):</label>
                <input type="number" name="endX" id="endX" class="form-control">
              </div>
              <div class="col">
                <label for="endY" class="form-label">切り抜き終了 Y (省略可能):</label>
                <input type="number" name="endY" id="endY" class="form-control">
              </div>
            </div>
            <button type="submit" class="btn btn-primary">検出</button>
          </form>
          <div id="result" class="mt-4"></div>
        </div>
      </body>
    </html>
    '''
    return render_template_string(form_html)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
