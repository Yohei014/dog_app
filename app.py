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
# TensorFlow設定（Render安定化）
# =========================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.config.set_visible_devices([], "GPU")

# =========================

app = Flask(__name__)

UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

img_size = 300

# =========================
# モデルロード
# =========================

model = load_model("dog_model.h5")

with open("dog_labels_ja.json", encoding="utf-8") as f:
    dog_labels = json.load(f)

with open("class_indices.json") as f:
    class_indices = json.load(f)

idx_to_class = {}

for folder_name, index in class_indices.items():

    if folder_name in dog_labels:
        idx_to_class[index] = dog_labels[folder_name]
    else:
        idx_to_class[index] = folder_name

# =========================
# 画像配信用ルート
# =========================

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# =========================
# 通常推論
# =========================

def predict(img_array):

    x = np.expand_dims(img_array, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x, verbose=0)

    return preds[0]

# =========================
# GradCAM
# =========================

def gradcam(img_array,img_path):

    last_conv_layer=None

    for layer in reversed(model.layers):

        if "conv" in layer.name:
            last_conv_layer=layer.name
            break

    grad_model=tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer).output,model.output]
    )

    x=np.expand_dims(img_array,axis=0)
    x=preprocess_input(x)

    with tf.GradientTape() as tape:

        conv_outputs,predictions=grad_model(x)
        class_idx=tf.argmax(predictions[0])
        loss=predictions[:,class_idx]

    grads=tape.gradient(loss,conv_outputs)
    pooled_grads=tf.reduce_mean(grads,axis=(0,1,2))

    conv_outputs=conv_outputs[0]
    heatmap=conv_outputs @ pooled_grads[...,tf.newaxis]
    heatmap=tf.squeeze(heatmap)

    heatmap=np.maximum(heatmap,0)

    if np.max(heatmap)!=0:
        heatmap/=np.max(heatmap)

    img=cv2.imread(img_path)
    h,w=img.shape[:2]

    heatmap=cv2.resize(heatmap,(w,h))
    heatmap=np.uint8(255*heatmap)
    heatmap=cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)

    superimposed_img=heatmap*0.4+img

    filename=f"gradcam_{uuid.uuid4().hex}.jpg"
    heatmap_path=os.path.join(app.config["UPLOAD_FOLDER"], filename)

    cv2.imwrite(heatmap_path,superimposed_img)

    return filename

# =========================
# ルート
# =========================

@app.route("/",methods=["GET","POST"])

def index():

    if request.method=="POST":

        file=request.files["file"]

        filename=f"{uuid.uuid4().hex}.jpg"

        filepath=os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(filepath)

        img=image.load_img(filepath,target_size=(img_size,img_size))
        img_array=image.img_to_array(img)

        pred=predict(img_array)

        top3=np.argsort(pred)[-3:][::-1]

        results=[]

        for i in top3:

            label=idx_to_class.get(int(i),"Unknown")

            results.append(
                (label,round(float(pred[i])*100,2))
            )

        gradcam_filename=gradcam(img_array,filepath)

        return render_template(
            "index.html",
            image=f"/uploads/{filename}",
            gradcam=f"/uploads/{gradcam_filename}",
            results=results
        )

    return render_template("index.html")

# =========================
# Render用
# =========================

if __name__ == "__main__":
    port = int(os.environ.get("PORT",8080))
    app.run(host="0.0.0.0",port=port)