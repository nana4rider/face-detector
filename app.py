import cv2
import io
from flask import Flask, request, send_file, jsonify
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
            "confidence": request.form.get("confidence"),  # 信頼度の閾値
            "minSize": request.form.get("minSize")         # 最小サイズ
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
