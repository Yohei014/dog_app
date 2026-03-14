import os
import json
import cv2
import uuid
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

# =========================
# TensorFlow メモリ節約設定
# =========================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TF_FORCE_CPU_MAX_VM_SIZE'] = '512'
tf.get_logger().setLevel("ERROR")

try:
    tf.config.set_visible_devices([], "GPU")
except:
    pass

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# =========================
# Flask初期化
# =========================
app = Flask(__name__)
UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

IMG_SIZE = 224

# =========================
# モデル・ラベルロード (エラー時に詳細を出す)
# =========================
print("--- Starting Model Load ---")

# ファイルが存在するか先にチェック
model_path = "dog_mixup_model.keras"
if not os.path.exists(model_path):
    print(f"ERROR: {model_path} not found in current directory!")

try:
    # compile=False でロード時のメモリ消費を最小化
    model = load_model(model_path, compile=False)
    print("SUCCESS: Model loaded.")
    
    with open("dog_labels_ja.json", encoding="utf-8") as f:
        dog_labels = json.load(f)
    with open("class_indices.json") as f:
        class_indices = json.load(f)
    
    idx_to_class = {int(v): dog_labels.get(k, k) for k, v in class_indices.items()}
    print("SUCCESS: Labels loaded.")
except Exception as e:
    print(f"FATAL ERROR DURING LOAD: {e}")
    # ここで model を None にしておき、実行時にチェックできるようにする
    model = None

# =========================
# 推論・GradCAM関数などは変更なし (前回のものと同じ)
# =========================
# ... (前回の get_gradcam 関数) ...

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # ボタンを押した時に model が存在するかチェック
        if model is None:
            return "サーバー起動時にモデルの読み込みに失敗しました。ログを確認してください。", 500
            
        try:
            # ... (以下、画像保存・推論処理) ...
            file = request.files["file"]
            filename = f"{uuid.uuid4().hex}.jpg"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            img = image.load_img(filepath, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = image.img_to_array(img)
            
            x = np.expand_dims(img_array, axis=0)
            x = preprocess_input(x)
            
            # ここで model.predict を実行
            pred = model.predict(x, verbose=0)[0]
            
            top3_idx = np.argsort(pred)[-3:][::-1]
            results = [(idx_to_class.get(int(i), "Unknown"), round(float(pred[i])*100, 2)) for i in top3_idx]

            # GradCAM (メモリ不足対策のため、失敗しても結果は出すように try-except)
            try:
                # get_gradcamの中身は前回の回答と同じものを想定
                gradcam_file = get_gradcam(img_array, filepath, model)
            except:
                gradcam_file = None

            return render_template(
                "index.html",
                image=f"/uploads/{filename}",
                gradcam=f"/uploads/{gradcam_file}" if gradcam_file else None,
                results=results
            )
        except Exception as e:
            print(f"POST Error: {e}")
            return f"実行エラー: {e}", 500

    return render_template("index.html")

# 以降省略 (前回の回答と同じ)