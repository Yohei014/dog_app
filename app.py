import os
# Keras 2 (Legacy) を強制使用
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['TF_FORCE_CPU_MAX_VM_SIZE'] = '512'

import json
import cv2
import uuid
import gc
import numpy as np

# インポートの順序と書き方を変更
import tensorflow as tf
from tensorflow import keras

# 必要な関数を直接参照
load_model = tf.keras.models.load_model
image = tf.keras.preprocessing.image
# EfficientNet用
preprocess_input = tf.keras.applications.efficientnet.preprocess_input

# CPUスレッド制限
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

app = tf.keras.preprocessing.image # 名前衝突回避のためのダミー（無視してOK）
from flask import Flask, render_template, request, send_from_directory
app = Flask(__name__)

UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

IMG_SIZE = 224
global_model = None
global_labels = None

def load_dog_ai():
    global global_model, global_labels
    if global_model is not None:
        return global_model, global_labels

    print("--- Starting Model Load (.h5) ---")
    model_path = "dog_mixup_model.h5"
    
    try:
        gc.collect()
        # compile=False でメモリ節約
        global_model = load_model(model_path, compile=False)
        
        with open("dog_labels_ja.json", encoding="utf-8") as f:
            labels_raw = json.load(f)
        with open("class_indices.json") as f:
            indices_raw = json.load(f)
        
        global_labels = {int(v): labels_raw.get(k, k) for k, v in indices_raw.items()}
        gc.collect()
        print("--- Model Loaded Successfully ---")
        return global_model, global_labels
    except Exception as e:
        print(f"FATAL ERROR DURING LOAD: {e}")
        return None, None

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        model, idx_to_class = load_dog_ai()
        if model is None:
            return "モデル読み込み失敗。ログを確認してください。", 500

        try:
            if "file" not in request.files: return render_template("index.html")
            file = request.files["file"]
            if file.filename == "": return render_template("index.html")

            filename = f"{uuid.uuid4().hex}.jpg"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            img = image.load_img(filepath, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = image.img_to_array(img)
            x = np.expand_dims(img_array, axis=0)
            x = preprocess_input(x)
            
            pred = model.predict(x, verbose=0)[0]
            top3_idx = np.argsort(pred)[-3:][::-1]
            results = [(idx_to_class.get(int(i), "不明"), round(float(pred[i])*100, 2)) for i in top3_idx]

            del img_array
            gc.collect()

            return render_template("index.html", image=f"/uploads/{filename}", results=results)
        except Exception as e:
            return f"Error: {e}", 500
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)