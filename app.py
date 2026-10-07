import json
import os
import gradio as gr
import numpy as np

# Force TensorFlow to run on CPU to avoid CUDA driver collisions
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

# 1. Path Resolution for Model and Threshold
MODEL_FILE = "improved_cnn_model.h5"
if not os.path.exists(MODEL_FILE) and os.path.exists("best_pneumonia_mobilenetv2.h5"):
    MODEL_FILE = "best_pneumonia_mobilenetv2.h5"

THRESHOLD_FILE = "best_threshold.json"
OPTIMAL_THRESHOLD = 0.50

if os.path.exists(THRESHOLD_FILE):
    try:
        with open(THRESHOLD_FILE, "r") as f:
            data = json.load(f)
            OPTIMAL_THRESHOLD = data.get("threshold", 0.50)
    except Exception as e:
        print(f"Notice: Could not parse threshold JSON ({e}), using 0.50.")

print(f"Loaded Decision Threshold: {OPTIMAL_THRESHOLD}")
print(f"Loading Model File: {MODEL_FILE}")

# 2. Load Model
model = tf.keras.models.load_model(MODEL_FILE)
print("Model successfully loaded!")

IMG_SIZE = (150, 150)

def predict_xray(image):
    if image is None:
        return None, "Please upload an X-ray image."

    # Preprocess Image
    img = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Model Inference
    raw_prob = float(model.predict(img_batch, verbose=0)[0][0])

    # Class Determination via Tuned Threshold
    prediction = "PNEUMONIA" if raw_prob >= OPTIMAL_THRESHOLD else "NORMAL"
    results = {"NORMAL": 1.0 - raw_prob, "PNEUMONIA": raw_prob}

    status_message = f"### Diagnostic Result: **{prediction}**\n"
    status_message += f"- **Pneumonia Probability:** `{raw_prob * 100:.2f}%`\n"
    status_message += f"- **Decision Threshold Used:** `{OPTIMAL_THRESHOLD}`\n\n"

    if prediction == "PNEUMONIA":
        status_message += "⚠️ **Note:** Radiological features consistent with Pneumonia detected."
    else:
        status_message += "✅ **Note:** Radiograph classified as Normal based on threshold."

    return results, status_message

# 3. Gradio Interface Construction
with gr.Blocks() as demo:
    gr.Markdown("# 🫁 AI Chest X-Ray Pneumonia Classifier")
    gr.Markdown(
        "Upload a chest radiograph to evaluate for Normal vs. Pneumonia features using the fine-tuned transfer learning model."
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload Chest Radiograph")
            submit_btn = gr.Button("Analyze Radiograph", variant="primary")

        with gr.Column(scale=1):
            output_label = gr.Label(num_top_classes=2, label="Probabilities")
            output_text = gr.Markdown(label="Diagnostic Summary")

    submit_btn.click(
        fn=predict_xray,
        inputs=[input_image],
        outputs=[output_label, output_text],
    )

if __name__ == "__main__":
    # Get port assigned by Render, defaulting to 7860 for local testing
    port = int(os.environ.get("PORT", 7860))

    # Launch with 0.0.0.0 binding and dynamic port mapping
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")
    )