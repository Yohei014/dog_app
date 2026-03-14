import os
import json
import uuid
import numpy as np
import cv2
from flask import Flask, render_template, request, send_from_directory
# tflite_runtime を使用（tensorflow本体は不要）
import tflite_runtime.interpreter as tflite

app = Flask(__name__)
UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# 起動時にインタープリター（推論エンジン）を準備
MODEL_PATH = "dog_mixup_model.tflite"
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# 入出力の情報を取得
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ラベルの読み込み
with open("dog_labels_ja.json", encoding="utf-8") as f:
    dog_labels = json.load(f)
with open("class_indices.json") as f:
    class_indices = json.load(f)
idx_to_class = {int(v): dog_labels.get(k, k) for k, v in class_indices.items()}

def preprocess_for_tflite(img_path):
    # OpenCVで読み込んで前処理（EfficientNetB0用: 224x224, 0-255）
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32)
    # EfficientNetの前処理（通常は tf.keras.applications.efficientnet.preprocess_input）
    # 実体は平均を引いて分散で割る等だが、B0は入力がそのままでも動くことが多い。
    # ここでは一般的な224x224の正規化を想定。
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return render_template("index.html")

        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # TFLiteで推論
        input_data = preprocess_for_tflite(filepath)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # 結果の取得
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Top-3の抽出
        top3_idx = np.argsort(output_data)[-3:][::-1]
        results = [(idx_to_class.get(int(i), "不明"), round(float(output_data[i])*100, 2)) for i in top3_idx]

        return render_template("index.html", image=f"/uploads/{filename}", results=results)

    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)