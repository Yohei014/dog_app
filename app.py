import os
import json
import cv2
import uuid
import gc
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

# =========================
# TensorFlow メモリ制限設定
# =========================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# TensorFlowがメモリを独占しないように制限
os.environ['TF_FORCE_CPU_MAX_VM_SIZE'] = '512'

# スレッド数を1に制限してメモリ消費を最小化
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

app = Flask(__name__)
UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

IMG_SIZE = 224
global_model = None
global_labels = None

# =========================
# モデルの遅延読み込み（メモリ節約版）
# =========================
def load_dog_ai():
    global global_model, global_labels
    if global_model is not None:
        return global_model, global_labels

    print("--- Starting Model Load (Memory Optimization Mode) ---")
    model_path = "dog_mixup_model.keras"
    
    try:
        # 読み込み前に不要なメモリを解放
        gc.collect()
        
        # compile=False は必須（学習用パラメータを読み込まないため軽量になる）
        global_model = load_model(model_path, compile=False)
        
        with open("dog_labels_ja.json", encoding="utf-8") as f:
            labels_raw = json.load(f)
        with open("class_indices.json") as f:
            indices_raw = json.load(f)
        
        global_labels = {int(v): labels_raw.get(k, k) for k, v in indices_raw.items()}
        
        # 読み込み後にも掃除
        gc.collect()
        print("--- Model Loaded Successfully ---")
        return global_model, global_labels
    except Exception as e:
        print(f"FATAL ERROR DURING LOAD: {e}")
        return None, None

# =========================
# GradCAMは一旦停止（コメントアウト）
# =========================
"""
def get_gradcam(img_array, img_path, model, scale=0.4):
    # メモリ節約のため、一旦無効化
    return None
"""

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # 判定時にモデルをロード
        model, idx_to_class = load_dog_ai()
        if model is None:
            return "モデルの初期化に失敗しました。ファイル名やメモリ制限を確認してください。", 500

        try:
            if "file" not in request.files:
                return render_template("index.html")
            
            file = request.files["file"]
            if file.filename == "":
                return render_template("index.html")

            # 保存
            filename = f"{uuid.uuid4().hex}.jpg"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # 前処理
            img = image.load_img(filepath, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = image.img_to_array(img)
            
            x = np.expand_dims(img_array, axis=0)
            x = preprocess_input(x)
            
            # 推論（CPU負荷を抑えるためにverbose=0）
            pred = model.predict(x, verbose=0)[0]
            
            # 結果の整理
            top3_idx = np.argsort(pred)[-3:][::-1]
            results = []
            for i in top3_idx:
                class_name = idx_to_class.get(int(i), f"Unknown({i})")
                score = round(float(pred[i]) * 100, 2)
                results.append((class_name, score))

            # メモリ解放を促進
            del img_array
            gc.collect()

            return render_template(
                "index.html",
                image=f"/uploads/{filename}",
                gradcam=None, # GradCAMは表示しない
                results=results
            )
        except Exception as e:
            print(f"POST Error: {e}")
            return f"実行エラー: {e}", 500

    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)