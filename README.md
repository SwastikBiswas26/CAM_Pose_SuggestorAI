# PosePerfect: AI-Powered Pose Guider & Suggester 📸🤖

**PosePerfect** is a python-based, real-time ML application designed to guide users on how to strike the perfect pose. Using computer vision and pose-matching algorithms, it detects the user's posture through their webcam, compares it to a library of reference poses, and provides real-time alignment scores and step-by-step instructions (e.g., *"Raise your left arm"* or *"Turn slightly right"*).

This project is ideal for photography enthusiasts, models, or anyone who feels awkward or unsure of how to pose in front of a camera.

---

## 🌟 Features
- **Real-Time Body Landmark Detection**: Powered by **MediaPipe Pose** to track 33 body skeleton joints at high FPS.
- **Pose Comparison Engine**: Compares the user's live posture with selected templates using joint angle calculations and spatial alignment.
- **Visual Phantom Overlay**: Renders a silhouette of the target pose directly on the webcam screen so the user can easily align themselves.
- **Interactive Feedback**: Dynamic match rating percentage and text-based directions.
- **Premium Web Dashboard**: A gorgeous, glassmorphic dark-mode web user interface built with HTML, CSS (Vanilla), and JavaScript.
- **Low-Latency Streaming**: Bidirectional communication via WebSockets for real-time video processing.

---

## 🛠️ Tech Stack
- **Backend**: Python, FastAPI, Uvicorn, MediaPipe, OpenCV, NumPy
- **Frontend**: HTML5, Vanilla CSS (Glassmorphism), JavaScript (Webcam Canvas API, WebSockets)

---

## 📁 Folder Structure
```text
NewProj/
├── backend/
│   ├── __init__.py          
│   ├── main.py              # FastAPI server & WebSocket endpoints
│   ├── pose_analyzer.py      # MediaPipe processing & similarity calculations
│   └── pose_library.py      # Reference poses landmarks definition
├── static/
│   ├── index.html           # Main frontend UI
│   ├── style.css            # Dark mode glassmorphic styling
│   └── app.js               # Canvas painting & WebSocket controller
├── assets/                  # Reference pose sample images
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/poseperfect.git
cd poseperfect
```

### 2. Set up Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python backend/main.py
```
Open your browser and navigate to `http://localhost:8000` to start using the app!
