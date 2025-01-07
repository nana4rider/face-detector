import cv2
import sys
import io
import os
from flask import Flask, request, send_file, jsonify
import numpy as np

app = Flask(__name__)

def detect_face_in_center(image_data):
    # Haarカスケードを使用した顔検出モデルをロード
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # 入力画像をデコード
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        print("入力画像を読み込めませんでした。")
        return {"error": "入力画像を読み込めませんでした。"}, 400

    # 中央部分の固定座標（環境変数から取得）
    img_height, img_width = image.shape[:2]
    x1 = int(os.getenv("X1", 0))
    y1 = int(os.getenv("Y1", 0))
    x2 = int(os.getenv("X2", img_width))
    y2 = int(os.getenv("Y2", img_height))

    # 中央部分を切り抜き
    center_image = image[y1:y2, x1:x2]

    # グレースケールに変換
    gray = cv2.cvtColor(center_image, cv2.COLOR_BGR2GRAY)

    # 顔検出パラメータ（環境変数から取得）
    scaleFactor = float(os.getenv("SCALE_FACTOR", 1.1))
    minNeighbors = int(os.getenv("MIN_NEIGHBORS", 1))
    minSize = int(os.getenv("MIN_SIZE", 30))

    # 顔を検出
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=(minSize, minSize))

    if len(faces) == 0:
        print("中央部分に顔が検出されませんでした。")
        return {"error": "中央部分に顔が検出されませんでした。"}, 404

    # 最大の顔を見つける
    largest_face = None
    max_area = 0

    for (x, y, w, h) in faces:
        area = w * h
        if area > max_area:
            max_area = area
            largest_face = (x, y, w, h)

    if largest_face is None:
        print("有効な顔が見つかりませんでした。")
        return {"error": "有効な顔が見つかりませんでした。"}, 404

    # 中央部分の顔の位置を元の画像の座標に変換
    x, y, w, h = largest_face
    x += x1
    y += y1

    # 顔部分を切り抜き
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

        return detect_face_in_center(image_data)

    except Exception as e:
        print(f"エラー: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
