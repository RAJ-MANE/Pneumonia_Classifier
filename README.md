# 🫁 AI Chest X-Ray Pneumonia Classifier

A Deep Learning web application for detecting **Pneumonia from Chest X-Ray images** using Transfer Learning with **MobileNetV2** and an interactive **Gradio** interface.

---

## 📌 Features
- **Binary Classification**: Identifies `NORMAL` vs. `PNEUMONIA` features from chest radiograph images.
- **Tuned Decision Threshold**: Uses an optimized decision threshold for balanced sensitivity and specificity.
- **Interactive Web UI**: Powered by Gradio for real-time inference.
- **Render & Cloud-Ready**: Includes dynamic port-binding and CPU execution configuration.

---

## 🚀 Quick Start (Local)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Gradio App
```bash
python app.py
```

Open your browser at `http://localhost:7860`.

---

## ☁️ Deployment on Render

| Setting | Value |
| --- | --- |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `python app.py` |
| **Instance Type** | `Free` |

---

## 📂 Project Structure
```text
├── app.py                     # Gradio Web Application
├── improved_cnn_model.h5      # Trained MobileNetV2 Model
├── best_threshold.json        # Optimized Classification Threshold
├── requirements.txt           # Python Dependencies
└── README.md                  # Project Documentation
```

---

## ⚠️ Disclaimer
*This tool is developed for educational and research purposes only and is not intended to replace professional medical diagnosis or clinical decision-making.*
