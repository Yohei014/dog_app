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
# TensorFlow設定（軽量化）
# =========================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")
try:
    tf.config.set_visible_devices([], "GPU")  # CPUのみ
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

IMG_SIZE = 224  # EfficientNetB0推論用

# =========================
# モデルロード
# =========================
print("Loading model...")
model = load_model("dog_mixup_model.keras", compile=False)
print("Model loaded")

# =========================
# ラベルロード
# =========================
with open("dog_labels_ja.json", encoding="utf-8") as f:
    dog_labels = json.load(f)

with open("class_indices.json") as f:
    class_indices = json.load(f)

idx_to_class = {int(v): dog_labels.get(k, k) for k, v in class_indices.items()}

# =========================
# 画像配信
# =========================
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# =========================
# 推論関数
# =========================
def predict(img_array):
    x = np.expand_dims(img_array, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x, verbose=0)
    return preds[0]

# =========================
# 軽量GradCAM
# =========================
def gradcam(img_array, img_path, scale=0.3):
    last_conv_layer = None
    for layer in reversed(model.layers):
        if "conv" in layer.name:
            last_conv_layer = layer.name
            break
    if last_conv_layer is None:
        return None

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer).output, model.output]
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

    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    heatmap = cv2.resize(heatmap, (int(w*scale), int(h*scale)))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (w, h))
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    filename = f"gradcam_{uuid.uuid4().hex}.jpg"
    heatmap_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    cv2.imwrite(heatmap_path, superimposed_img)
    return filename

# =========================
# メインページ
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html")
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html")
        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)

        pred = predict(img_array)
        top3 = np.argsort(pred)[-3:][::-1]
        results = [(idx_to_class.get(int(i), "Unknown"), round(float(pred[i])*100, 2)) for i in top3]

        gradcam_file = gradcam(img_array, filepath)

        return render_template(
            "index.html",
            image=f"/uploads/{filename}",
            gradcam=f"/uploads/{gradcam_file}" if gradcam_file else None,
            results=results
        )
    return render_template("index.html")

# =========================
# Renderデプロイ用
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)