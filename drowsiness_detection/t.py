import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model
import os

# === Load model ===
model = load_model("drowsiness_detection\DDD\my_model.keras")  # Update path if needed

# === Define binary classification labels ===
def interpret_prediction(pred):
    label = "drowsy" if pred >= 0.5 else "awake"
    prob = float(pred) if pred >= 0.5 else 1.0 - float(pred)
    return label, prob

# === Parameters ===
sequence_length = 5
frame_size = (128, 128)

# === Prepare frame queue ===
frame_queue = deque(maxlen=sequence_length)

# === Input/output video paths ===
input_path = "media\daraset (1).mp4"
output_path = r"media\output22.avi"
os.makedirs("media", exist_ok=True)

# === Open video and prepare writer ===
cap = cv2.VideoCapture(input_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# === Inference loop ===
frame_count = 0
current_label = ""
current_confidence = 0.0
raw_sigmoid = 0.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Preprocess current frame
    resized = cv2.resize(frame, frame_size)
    normalized = resized / 255.0
    frame_queue.append(normalized)

    # Predict only every 5 frames and if queue is full
    if len(frame_queue) == sequence_length and frame_count % 5 == 0:
        sequence = np.expand_dims(np.array(frame_queue), axis=0)  # (1, 5, 128, 128, 3)
        pred = model.predict(sequence, verbose=0)[0][0]
        raw_sigmoid = pred
        current_label, current_confidence = interpret_prediction(pred)

    # Draw current prediction (from last prediction step)
    if current_label:
        text = f"{current_label} ({current_confidence:.2f})"
        color = (0, 0, 255) if current_label == "drowsy" else (0, 255, 0)
        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Raw Sigmoid: {raw_sigmoid:.4f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Write frame to output video
    out.write(frame)

# === Release everything ===
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Video saved to media\\output.avi")
