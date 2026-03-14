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
# メモリを一度に確保させない設定
os.environ['TF_FORCE_CPU_MAX_VM_SIZE'] = '512'
tf.get_logger().setLevel("ERROR")

try:
    tf.config.set_visible_devices([], "GPU")
except:
    pass

# スレッド数を制限してメモリ消費を抑える
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
# モデル・ラベルロード (起動時に1回)
# =========================
print("Loading model and labels...")
try:
    # compile=False でロード時間を短縮しメモリを節約
    model = load_model("dog_mixup_model.keras", compile=False)
    
    with open("dog_labels_ja.json", encoding="utf-8") as f:
        dog_labels = json.load(f)
    with open("class_indices.json") as f:
        class_indices = json.load(f)
    
    # インデックスを整数に変換してマッピング
    idx_to_class = {int(v): dog_labels.get(k, k) for k, v in class_indices.items()}
    print("Model and labels loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR DURING STARTUP: {e}")

# =========================
# 画像配信
# =========================
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# =========================
# 軽量GradCAM
# =========================
def get_gradcam(img_array, img_path, model, scale=0.4):
    try:
        # 最後の畳み込み層を特定
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            if "conv" in layer.name:
                last_conv_layer_name = layer.name
                break
        
        if not last_conv_layer_name:
            return None

        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )

        x = np.expand_dims(img_array, axis=0)
        x = preprocess_input(x)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(x)
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)

        # 画像読み込みと合成
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        heatmap_resized = cv2.resize(heatmap.numpy(), (w, h))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        
        superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

        filename = f"gradcam_{uuid.uuid4().hex}.jpg"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        cv2.imwrite(save_path, superimposed_img)
        return filename
    except Exception as e:
        print(f"GradCAM error: {e}")
        return None

# =========================
# メインページ
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            if "file" not in request.files:
                return render_template("index.html")
            
            file = request.files["file"]
            if file.filename == "":
                return render_template("index.html")

            # 画像保存
            filename = f"{uuid.uuid4().hex}.jpg"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # 判定用前処理
            img = image.load_img(filepath, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = image.img_to_array(img)
            
            # 推論
            x = np.expand_dims(img_array, axis=0)
            x = preprocess_input(x)
            pred = model.predict(x, verbose=0)[0]
            
            # Top-3 結果作成
            top3_idx = np.argsort(pred)[-3:][::-1]
            results = []
            for i in top3_idx:
                class_name = idx_to_class.get(int(i), f"Unknown({i})")
                score = round(float(pred[i]) * 100, 2)
                results.append((class_name, score))

            # GradCAM生成
            gradcam_file = get_gradcam(img_array, filepath, model)

            return render_template(
                "index.html",
                image=f"/uploads/{filename}",
                gradcam=f"/uploads/{gradcam_file}" if gradcam_file else None,
                results=results
            )
        except Exception as e:
            print(f"POST Error: {e}")
            return f"エラーが発生しました: {e}", 500

    return render_template("index.html")

if __name__ == "__main__":
    # Renderのポート設定に対応
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)