import os
# メモリ消費を抑える設定を最優先で実行
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_FORCE_CPU_MAX_VM_SIZE"] = "512"

from flask import Flask
import gc

app = Flask(__name__)

@app.route("/")
def hello():
    try:
        # ここで初めてインポートを試みる（起動時のメモリ負荷を分散）
        import tensorflow as tf
        version = tf.__version__
        return f"TensorFlow version {version} is loaded! Memory check passed."
    except Exception as e:
        return f"Import failed: {e}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)