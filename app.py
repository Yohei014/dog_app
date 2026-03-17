import os
import json
import uuid
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from PIL import Image

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "/tmp/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# 必要な時だけ読み込む（メモリ節約）
interpreter = None
idx_to_class = None

def get_resources():
    global interpreter, idx_to_class
    if interpreter is None:
        import tflite_runtime.interpreter as tflite
        # モデル名は学習コードで保存した名前に合わせてください
        interpreter = tflite.Interpreter(model_path="dog_model.tflite")
        interpreter.allocate_tensors()
        
        # ラベルデータの読み込み（これらもGitHubに必要です）
        with open("dog_labels_ja.json", encoding="utf-8") as f:
            dog_labels = json.load(f)
        with open("class_indices.json") as f:
            class_indices = json.load(f)
        idx_to_class = {int(v): dog_labels.get(k, k) for k, v in class_indices.items()}
    return interpreter, idx_to_class

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            interp, labels = get_resources()
            file = request.files.get("file")
            if not file or file.filename == "":
                return render_template("index.html")

            # 保存
            filename = f"{uuid.uuid4().hex}.jpg"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Pillowで画像処理（OpenCVより軽量）
            img = Image.open(filepath).convert('RGB')
            img = img.resize((224, 224))
            input_data = np.expand_dims(np.array(img).astype(np.float32), axis=0)

            # 推論
            input_details = interp.get_input_details()
            output_details = interp.get_output_details()
            interp.set_tensor(input_details[0]['index'], input_data)
            interp.invoke()
            output_data = interp.get_tensor(output_details[0]['index'])[0]
            
            # 結果取得
            top3_idx = np.argsort(output_data)[-3:][::-1]
            results = [(labels.get(int(i), "unknown"), round(float(output_data[i])*100, 2)) for i in top3_idx]

            return render_template("index.html", image=f"/uploads/{filename}", results=results)
        except Exception as e:
            return f"Error: {e}", 500
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))