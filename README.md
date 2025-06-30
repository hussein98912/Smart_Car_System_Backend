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
- 🎵 Smart car assistant with music and alarm APIs
- ⚙️ Custom data-to-response pipeline that processes AI model outputs and formats them for API responses


---

## 📁 Project Structure

```
smart_car_backend/
│
│
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
├── README                    # Project documentation 
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

| Model Type               | File Name                |
|--------------------------|--------------------------|
| Vehicle Detection (YOLO) | `best(1).pt`             |
| Depth Estimation (MiDaS) | `dpt_large_384.pt`       |
| Sign/Road Classifier     | `resnet_model.pt`        |
| Drowsiness Detection     | `drowsiness_cnn_lstm.h5` |

---

**Notes:**  
- The **Depth Estimation model (`dpt_large_384.pt`)** can be downloaded from the official MiDaS repository here:  
  [https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_large_384.pt](https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_large_384.pt)  
- The **Vehicle Detection (YOLO)** and **Drowsiness Detection** models are maintained in separate repositories:  
  - Vehicle Detection Model Repository: [https://github.com/hussein98912/Vehicle-Detection-Model](https://github.com/hussein98912/Vehicle-Detection-Model)  
  - Drowsiness Detection Model Repository: [https://github.com/hussein98912/Drowsiness-Detection-Model](https://github.com/hussein98912/Drowsiness-Detection-Model)  
- The **Sign/Road Classifier (`resnet_model.pt`)** is included within this project.

---

## 📦 Sample API Endpoints

Vehicle Detection, Signs, Road Status, and Video Processing

| Endpoint             | Description                                |
| -------------------- | ------------------------------------------ |
| `/close_vehicles/`   | Returns vehicles detected close to the car |
| `/detected_signals/` | Returns detected traffic signs             |
| `/road_status/`      | Provides current road condition            |
| `/start_processing/` | Starts video processing pipeline           |

Drowsiness Detection

| Endpoint                  | Description                         |
| ------------------------- | ----------------------------------- |
| `/drowsiness/api/detect/` | Returns current drowsiness status   |
| `/drowsiness/api/start/`  | Starts drowsiness detection process |

Driving Pattern Prediction

| Endpoint                    | Description                           |
| --------------------------- | ------------------------------------- |
| `/predict-driving-pattern/` | Predicts the driving behavior pattern |

Smart Assistant 

| Endpoint                           | Description                          |
| ---------------------------------- | ------------------------------------ |
| `/assistant/api/start-wake-word/`  | Starts wake word listening           |
| `/assistant/api/get-intent/`       | Retrieves intent from voice commands |
| `/assistant/api/wake-word-status/` | Returns wake word listening status   |

---

⚙️ Data-to-Response Pipeline
The backend processes raw outputs from multiple AI models through a custom pipeline that:

Parses and filters model outputs

Integrates data from different models (e.g., vehicle detection + depth estimation)

Formats and structures the data as clean JSON responses

Supports real-time response for the smart car assistant APIs

This pipeline ensures consistency, reliability, and speed in the backend responses.


---

## 🧑‍💻 Author

**Hussein Slman**  
AI Engineer | Computer Vision | NLP  
GitHub: [@hussein98912](https://github.com/hussein98912)

---

## 📜 License

MIT License 
