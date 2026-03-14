import os
# メモリ設定を最優先
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_FORCE_CPU_MAX_VM_SIZE"] = "512"

import json
import gc
import numpy as np
import uuid
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)
UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# グローバル変数を空で用意
global_model = None
global_labels = None

def load_model_on_demand():
    global global_model, global_labels
    if global_model is not None:
        return global_model, global_labels

    # 判定ボタンが押された時だけ、ここで重いライブラリとモデルを読み込む
    import tensorflow as tf
    
    # スレッド制限でメモリバーストを防止
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    
    model_path = "dog_mixup_model.h5"
    
    # モデルロード
    global_model = tf.keras.models.load_model(model_path, compile=False)
    
    # ラベルロード
    with open("dog_labels_ja.json", encoding="utf-8") as f:
        labels_raw = json.load(f)
    with open("class_indices.json") as f:
        indices_raw = json.load(f)
    
    global_labels = {int(v): labels_raw.get(k, k) for k, v in indices_raw.items()}
    
    gc.collect()
    return global_model, global_labels

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # モデルを呼び出す（初回のみロードが走る）
            model, idx_to_class = load_model_on_demand()
            
            # 画像処理ライブラリもここでインポート
            import tensorflow as tf
            from tensorflow.keras.preprocessing import image
            from tensorflow.keras.applications.efficientnet import preprocess_input

            file = request.files.get("file")
            if not file or file.filename == "":
                return render_template("index.html")

            filename = f"{uuid.uuid4().hex}.jpg"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # 推論処理
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            x = np.expand_dims(img_array, axis=0)
            x = preprocess_input(x)
            
            pred = model.predict(x, verbose=0)[0]
            top3_idx = np.argsort(pred)[-3:][::-1]
            results = [(idx_to_class.get(int(i), "不明"), round(float(pred[i])*100, 2)) for i in top3_idx]

            # メモリ掃除
            del img_array
            gc.collect()

            return render_template("index.html", image=f"/uploads/{filename}", results=results)
        
        except Exception as e:
            return f"判定エラーが発生しました: {e}", 500

    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)