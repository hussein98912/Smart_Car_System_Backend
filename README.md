# 🚗 Smart Car System - Backend

A Django-based backend system for a Smart Car assistant. It integrates multiple AI models to provide real-time services such as:

- Vehicle detection and classification
- Depth estimation from monocular images
- Traffic sign and road damage recognition
- Driver drowsiness detection
- Smart assistant NLP interface
- Integration with Google Maps and music/alarm systems

---

## 🛠 Features

- 🔍 **YOLOv8** for vehicle detection
- 📏 **MiDaS** for depth estimation
- 🧠 **PyTorch CNN** for road condition and traffic sign classification
- 💤 **Drowsiness Detection** using TimeDistributed CNN + LSTM
- 🎤 **Voice Command Processing** using NLP (Speech-to-Text / Text-to-Speech)
- 🗺️ Integration with Google Maps API
- 🎵 Smart car assistant with music and alarm APIs

---

## 📁 Project Structure

```
smart_car_backend/
│
├── api/                      # REST API endpoints
├── drive_class/              # Vehicle classification module
├── drowsiness_detection/     # Drowsiness detection logic
├── front_camera_detection/   # Object detection & depth estimation
├── media/                    # Uploaded media or output files
├── midas/                    # Depth estimation models/utilities
├── smart_assistance/         # Smart assistant (NLP, voice commands)
├── smart_car_backend/        # Main Django project folder
├── venv/                     # Virtual environment (excluded by .gitignore)
│
├── .gitignore                # Git ignore rules
├── db.sqlite3                # SQLite database
├── LICENSE                   # License file
├── manage.py                 # Django management script
├── README                    # Project documentation (this file)
├── requirements              # Dependencies list
├── wake_word_status          # Wake word tracking file
```

---

## 🚀 Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run migrations
python manage.py migrate

# Run server
python manage.py runserver
```

---

## 🧠 AI Models

Make sure to download the following models and place them in the correct `models/` or related folders:

| Model Type               | File Name                     |
|--------------------------|-------------------------------|
| Vehicle Detection (YOLO) | `best(1).pt`                  |
| Depth Estimation (MiDaS) | `dpt_large_384.pt`            |
| Sign/Road Classifier     | `resnet_model.pt`             |
| Drowsiness Detection     | `drowsiness_cnn_lstm.h5`      |

> 🔗 You can host model files using Google Drive, Hugging Face, or similar platforms and add the links here.

---

## 📦 Sample API Endpoints

| Endpoint             | Description                    |
|----------------------|--------------------------------|
| `/detect_vehicle/`   | Returns detected vehicles      |
| `/estimate_depth/`   | Returns depth map              |
| `/classify_signs/`   | Returns detected traffic signs |
| `/detect_drowsiness/`| Detects drowsy state           |
| `/nlp/`              | Handles smart voice commands   |

---

## 🧑‍💻 Author

**Hussein Slman**  
AI Engineer | Computer Vision | NLP  
GitHub: [@hussein98912](https://github.com/hussein98912)

---

## 📜 License

MIT License — free to use and modify
