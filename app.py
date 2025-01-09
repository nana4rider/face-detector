import cv2
import io
from flask import Flask, request, send_file, jsonify
import numpy as np

app = Flask(__name__)

def detect_face_with_resize(image_data, params):
    # Haarカスケードを使用した顔検出モデルをロード
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # 入力画像をデコード
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        print("入力画像を読み込めませんでした。")
        return {"error": "入力画像を読み込めませんでした。"}, 400

    # 縮小スケール（デフォルト: 0.5）
    scale = float(params.get("scale") or 0.5)

    # 縮小処理（scale=1の場合はスキップ）
    if scale < 1:
        small_image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        gray_small = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_small = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 顔検出パラメータ
    scaleFactor = float(params.get("scaleFactor") or 1.1)
    minNeighbors = int(params.get("minNeighbors") or 2)
    minSize = int(params.get("minSize") or 80)

    # 顔を検出
    faces_small = face_cascade.detectMultiScale(
        gray_small,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=(int(minSize * scale), int(minSize * scale))
    )

    if len(faces_small) == 0:
        print("顔が検出されませんでした。")
        return {"error": "顔が検出されませんでした。"}, 404

    # 検出された顔の座標を元のサイズに変換（縮小時のみ）
    if scale < 1:
        faces_original = [(int(x / scale), int(y / scale), int(w / scale), int(h / scale)) for (x, y, w, h) in faces_small]
    else:
        faces_original = faces_small

    # 最大の顔を見つける
    largest_face = max(faces_original, key=lambda f: f[2] * f[3])  # 面積で最大の顔を選択

    # 顔部分を切り抜き
    x, y, w, h = largest_face
    face_image = image[y:y+h, x:x+w]

    # バイナリデータを返却
    _, buffer = cv2.imencode('.jpg', face_image)
    return send_file(io.BytesIO(buffer), mimetype='image/jpeg'), 200

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "ファイルがアップロードされていません。"}), 400

        file = request.files['file']
        image_data = file.read()

        # リクエストパラメータを取得
        params = {
            "scale": request.form.get("scale"),
            "scaleFactor": request.form.get("scaleFactor"),
            "minNeighbors": request.form.get("minNeighbors"),
            "minSize": request.form.get("minSize")
        }

        # 顔検出処理
        return detect_face_with_resize(image_data, params)

    except Exception as e:
        print(f"エラー: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
