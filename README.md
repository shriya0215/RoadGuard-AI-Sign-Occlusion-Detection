# RoadGuard AI – Intelligent Road Sign Occlusion Detection System

[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00.svg?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-000000.svg?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.1-5C3EE8.svg?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-00D9FF.svg?style=flat)]()

RoadGuard AI is a research-grade, dual-output Convolutional Neural Network (CNN) and interactive web application designed to evaluate and mitigate road sign perception failures in autonomous driving systems. It detects traffic signs across 43 classes while simultaneously predicting binary occlusion (Clear vs. Occluded) caused by environmental obstacles such as tree foliage, mud splatters, and graffiti vandalism.

---

## 1. Problem Statement
Autonomous vehicles rely heavily on computer vision to interpret traffic regulations (e.g., speed limits, stop signs, yield warnings). However, in real-world driving environments, traffic signs frequently suffer from partial or severe visual occlusions:
- Overgrown tree branches and leaves blocking sign text.
- Road spray, dirt, and mud obscuring critical pictograms.
- Urban vandalism, stickers, and graffiti spray paint altering sign geometries.

Standard single-head classification models often produce overconfident, erroneous predictions when evaluating occluded signs, creating dangerous hazards for trajectory planning. RoadGuard AI addresses this vulnerability through a joint dual-head architecture that classifies the sign while assessing physical occlusion probability, triggering fail-safe safety policies.

---

## 2. Key Objectives
1. **Multi-Task Latent Feature Sharing:** Learn robust visual representations through a shared CNN backbone.
2. **Dual-Head Output:**
   - **Head 1:** Traffic Sign Classification across 43 GTSRB classes (Softmax).
   - **Head 2:** Binary Occlusion Assessment (0 = Clear, 1 = Occluded, Softmax).
3. **Synthetic Occlusion Modeling:** Realistic procedural simulation of Leaves, Mud, and Graffiti with intensity scaling (10% to 95%).
4. **Safety Policy Framework:** Fuses sign confidence and occlusion probability into tiered autonomous alerts:
   - **Normal Detection:** Clear sign with high confidence (&ge; 60%).
   - **Caution – Partially Blocked:** Occlusion flagged by vision head.
   - **Uncertain – Human Verification Required:** Low confidence detection requiring operator fallback or LiDAR fusion.
5. **Interactive Autonomous Research Cockpit:** Modern glassmorphic web dashboard with real-time inference, simulator lab, before/after degradation comparisons, and Chart.js metrics.

---

## 3. Technology Stack

- **Machine Learning & Deep Learning:**
  - Python 3.12
  - TensorFlow 2.21 / Keras 3.15
  - OpenCV (cv2)
  - NumPy & Pandas
  - Scikit-learn
  - Matplotlib & Seaborn
- **Backend Service:**
  - Flask 3.1
  - RESTful JSON API
  - Werkzeug Secure I/O
- **Frontend Dashboard:**
  - Semantic HTML5 & Modern CSS3
  - Glassmorphic UI with Deep Navy (`#071426`) & Electric Cyan (`#00D9FF`) theme
  - Vanilla ES6 JavaScript (No bloated frameworks)
  - Chart.js for interactive metrics

---

## 4. Custom CNN Architecture

```
                  INPUT IMAGE (64×64×3 RGB)
                              │
                    ┌───────────────────┐
                    │   CONV BLOCK 1    │
                    │ Conv2D (32, 3×3)  │
                    │ BatchNormalization│
                    │       ReLU        │
                    │ Conv2D (32, 3×3)  │
                    │ BatchNormalization│
                    │       ReLU        │
                    │  MaxPooling2D(2×2)│
                    └─────────┬─────────┘
                              │
                    ┌───────────────────┐
                    │   CONV BLOCK 2    │
                    │ Conv2D (64, 3×3)  │
                    │ BatchNormalization│
                    │       ReLU        │
                    │ Conv2D (64, 3×3)  │
                    │ BatchNormalization│
                    │       ReLU        │
                    │  MaxPooling2D(2×2)│
                    └─────────┬─────────┘
                              │
                    ┌───────────────────┐
                    │   CONV BLOCK 3    │
                    │ Conv2D (128, 3×3) │
                    │ BatchNormalization│
                    │       ReLU        │
                    │ Conv2D (128, 3×3) │
                    │ BatchNormalization│
                    │       ReLU        │
                    │  MaxPooling2D(2×2)│
                    └─────────┬─────────┘
                              │
                 GLOBAL + SPATIAL CONCATENATION
                 (GAP + Flatten → 8,320 features)
                              │
                 SHARED DENSE (256, ReLU) + BN
                              │
               ┌──────────────┴──────────────┐
               │                             │
        TRAFFIC SIGN HEAD             OCCLUSION HEAD
        Dense (128, ReLU)             Dense (64, ReLU)
        BatchNormalization            BatchNormalization
        Dropout (0.20)                Dropout (0.20)
        Dense (43, Softmax)           Dense (2, Softmax)
```

### Combined Loss Function:
$$\mathcal{L}_{\text{total}} = 1.0 \times \mathcal{L}_{\text{sign}} (\text{CE}) + 1.0 \times \mathcal{L}_{\text{occlusion}} (\text{BCE})$$

Both heads are trained jointly using the Adam optimizer with learning rate reduction on plateau and checkpointing.

---

## 5. Project Folder Structure

```
RoadGuard-AI/
├── app.py                      # Flask Application Server
├── requirements.txt            # Python Dependencies
├── README.md                   # Complete Documentation
│
├── dataset/
│   ├── README.md               # Dataset details & download guide
│   └── Images/                 # 43 GTSRB sign class folders
│
├── models/
│   └── roadguard_cnn.keras     # Trained dual-head Keras CNN model
│
├── training/
│   ├── __init__.py
│   ├── data_loader.py          # Leak-free splitting & GTSRB loader
│   ├── occlusion_generator.py  # Realistic Leaves, Mud, Graffiti synth
│   ├── cnn_model.py            # Custom Dual-Output CNN definition
│   ├── train.py                # Joint training script with callbacks
│   └── evaluate.py             # Test split evaluation & plot generator
│
├── backend/
│   ├── __init__.py
│   ├── routes.py               # REST API endpoints
│   └── preprocessing.py        # Image validation, scaling, safety rules
│
├── frontend/
│   ├── index.html              # Main Cockpit (All modules)
│   ├── dashboard.html          # Dashboard Module
│   ├── scanner.html            # AI Scanner Module
│   ├── occlusion_lab.html      # Occlusion Simulator Module
│   ├── comparison.html         # Before vs After Analysis
│   ├── performance.html        # Model Performance Module
│   │
│   ├── css/
│   │   └── style.css           # Futuristic glassmorphism styling
│   │
│   └── js/
│       ├── main.js             # Navigation & health polling
│       ├── scanner.js          # Inference & preview logic
│       ├── simulator.js        # Lab & comparison controller
│       └── charts.js           # Chart.js visualization logic
│
├── results/
│   ├── metrics.json            # Actual test split evaluation metrics
│   ├── history.json            # Epoch-by-epoch loss & accuracy history
│   ├── training_history.png    # Loss & accuracy curves plot
│   ├── confusion_matrix.png    # Occlusion confusion matrix heatmap
│   └── classification_report.txt # Detailed precision/recall report
│
└── uploads/                    # Temporary uploaded images
```

---

## 6. Installation & Quick Start

### 1. Clone & Set Up Virtual Environment
```bash
# Clone the repository
cd RoadGuard-AI

# Create virtual environment (Python 3.12 recommended)
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the CNN Model
```bash
python training/train.py --epochs 10 --batch-size 32
```
This script will:
- Load/generate the 43 GTSRB class dataset.
- Perform a strict leak-free split (Train, Val, Test).
- Generate synthetic occluded variants (Leaves, Mud, Graffiti).
- Jointly train the dual-output CNN.
- Save the best model to `models/roadguard_cnn.keras`.
- Save plots and metrics to `results/`.

### 3. Launch the Web Application
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 7. REST API Documentation

### `POST /api/predict`
Runs dual-output CNN inference on an uploaded image.
- **Payload:** Multipart form-data with `image` file OR JSON with base64 `image`.
- **Response:**
  ```json
  {
    "success": true,
    "traffic_sign": {
      "class_id": 1,
      "name": "Speed limit (30km/h)",
      "confidence": 0.985,
      "confidence_percent": 98.5,
      "top_candidates": [...]
    },
    "occlusion": {
      "is_occluded": false,
      "status": "Clear",
      "confidence": 0.992,
      "probabilities": { "clear": 0.992, "occluded": 0.008 }
    },
    "safety": {
      "status": "Clear Sign",
      "severity": "success",
      "alert_title": "NORMAL DETECTION",
      "recommendation": "Road sign 'Speed limit (30km/h)' is clearly visible with 98.5% confidence."
    }
  }
  ```

### `POST /api/simulate-occlusion`
Applies procedural synthetic occlusion to an image.
- **Parameters:** `image`, `occlusion_type` (`leaves` | `mud` | `graffiti`), `intensity` (0.1 to 0.95).
- **Response:** Base64 encoded original and occluded images with metadata.

### `POST /api/compare`
Runs comparative dual inference on both clean and occluded versions of an image, returning the confidence delta and flagging adversarial perception flips.

### `GET /api/metrics`
Returns actual evaluated test metrics (`sign_accuracy`, `occlusion_precision`, `occlusion_recall`, `occlusion_f1`, confusion matrix).

### `GET /api/history`
Returns epoch-by-epoch loss and accuracy numbers for Chart.js rendering.

### `GET /api/health`
Returns system status, active CNN model path, and images processed count.

---

## 8. Safety Evaluation & Policy Rules

RoadGuard AI implements a three-tier deterministic perception safety policy:
| Condition | Tier & Severity | Recommended Autonomous Action |
| :--- | :--- | :--- |
| Sign is Clear AND Confidence &ge; 60% | **NORMAL DETECTION** (Success) | Nominal trajectory execution. Standard speed and lane control. |
| Occlusion Head flags Occluded (&ge; 50%) | **CAUTION – PARTIALLY BLOCKED** (Warning) | Perception warning logged. Speed reduced. Priority given to GPS/map priors. |
| Confidence &lt; 60% OR Ambiguity | **UNCERTAIN – VERIFICATION REQUIRED** (Danger) | Disengage high-speed autonomy. Prompt driver intervention or LiDAR fail-safe. |

*Disclaimer: RoadGuard AI is an academic research software system. It is not certified under ISO 26262 ASIL-D for direct vehicular actuator steering or braking.*
