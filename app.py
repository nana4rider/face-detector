import cv2
import io
from flask import Flask, request, send_file, jsonify, render_template_string
import numpy as np
import mediapipe as mp

app = Flask(__name__)

def detect_face_with_resize(image_data, params):
    # Mediapipeの顔検出を初期化
    mp_face_detection = mp.solutions.face_detection
    min_confidence = float(params.get("confidence") or 0.5)
    min_size = int(params.get("minSize") or 0)  # 最小サイズ（ピクセル単位）
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=min_confidence)

    # 入力画像をデコード
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        print("入力画像を読み込めませんでした。")
        return {"error": "入力画像を読み込めませんでした。"}, 400

    # Mediapipeで顔検出
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.detections:
        print("顔が検出されませんでした。")
        return {"error": "顔が検出されませんでした。"}, 404

    # 検出された顔の座標を取得
    faces_original = []
    ih, iw, _ = image.shape
    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
        # 最小サイズ以上の顔のみ追加
        if w >= min_size and h >= min_size:
            faces_original.append((x, y, w, h))

    # フィルタリング後に顔がない場合
    if not faces_original:
        print("指定されたサイズ以上の顔が検出されませんでした。")
        return {"error": "指定されたサイズ以上の顔が検出されませんでした。"}, 404

    # 最大の顔を見つける
    largest_face = max(faces_original, key=lambda f: f[2] * f[3])  # 面積で最大の顔を選択

    # 顔部分を切り抜き
    x, y, w, h = largest_face
    face_image = image[y:y+h, x:x+w]

    # バイナリデータを返却
    _, buffer = cv2.imencode('.jpg', face_image)
    return buffer, (w, h)

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "ファイルがアップロードされていません。"}), 400

        file = request.files['file']
        image_data = file.read()

        # リクエストパラメータを取得
        params = {
            "confidence": request.form.get("confidence"),  # 信頼度の閾値
            "minSize": request.form.get("minSize")         # 最小サイズ
        }

        # 顔検出処理
        result = detect_face_with_resize(image_data, params)
        if isinstance(result, tuple):
            buffer, dimensions = result
            response = send_file(io.BytesIO(buffer), mimetype='image/jpeg')
            response.headers['X-Image-Width'] = dimensions[0]
            response.headers['X-Image-Height'] = dimensions[1]
            return response
        else:
            return jsonify(result)

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
        <title>顔検出テストフォーム</title>
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
                resultDiv.innerHTML = `<img src="${imgUrl}" alt="Detected Face"><p>Width: ${width}px, Height: ${height}px</p>`;
              } else {
                const error = await response.json();
                resultDiv.innerHTML = `<p style="color: red;">エラー (${response.status}): ${error.error}</p>`;
              }
            } catch (err) {
              document.getElementById('result').innerHTML = `<p style="color: red;">エラー: ${err.message}</p>`;
            }
          }
        </script>
      </head>
      <body>
        <h1>顔検出テストフォーム</h1>
        <form action="/detect" method="post" enctype="multipart/form-data" onsubmit="handleSubmit(event)">
          <label for="file">画像ファイル:</label>
          <input type="file" name="file" id="file" accept="image/*" required><br><br>

          <label for="confidence">信頼度閾値 (0.0 - 1.0):</label>
          <input type="number" name="confidence" id="confidence" step="0.1" min="0" max="1" value="0.5"><br><br>

          <label for="minSize">最小サイズ (ピクセル):</label>
          <input type="number" name="minSize" id="minSize" value="0"><br><br>

          <button type="submit">検出</button>
        </form>
        <div id="result" style="margin-top: 20px;"></div>
      </body>
    </html>
    '''
    return render_template_string(form_html)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
