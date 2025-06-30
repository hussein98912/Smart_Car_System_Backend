from ultralytics import YOLO
import cv2
import os

# Load both models
vd_model = YOLO(r"front_camera_detection\models\best (1).pt")  # Vehicle detection
tl_model = YOLO(r"front_camera_detection\models\best.pt")      # Traffic light detection

# Open input video
cap = cv2.VideoCapture(r"media\light.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Ensure media directory exists
os.makedirs("media", exist_ok=True)

# Output writer
out = cv2.VideoWriter(r"media\output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # === Vehicle Detection ===
    vd_results = vd_model(frame, conf=0.3)[0]

    for box in vd_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{vd_model.names[cls]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

    # === Traffic Light Detection ===
    tl_results = tl_model(frame, conf=0.3)[0]

    for box in tl_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{tl_model.names[cls]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Write frame to output
    out.write(frame)

cap.release()
out.release()
print("✅ Output video saved at: media/output.mp4")
