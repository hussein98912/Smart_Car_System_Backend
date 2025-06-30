import cv2
import numpy as np
import os
import collections
import tensorflow as tf
import time

# === الإعدادات العامة ===
sequence_length = 5
frame_size = (128, 128)
pred_interval = 0.5
status_smoothing_window = 10

# === المسارات ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'my_model.keras')
status_path = os.path.join(BASE_DIR, 'status')

# === تحميل النموذج ===
model = tf.keras.models.load_model(model_path)

# === تحسين الإضاءة + توضيح الصورة ===
def preprocess_image(image):
    # تحسين التباين باستخدام CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # توضيح الحواف
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    # تقليل الضوضاء
    denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 10, 10, 7, 21)

    return denoised

# === الكاميرا ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)  # محاولة لرفع السطوع

# === المتغيرات ===
frame_buffer = collections.deque(maxlen=sequence_length)
status_history = collections.deque(maxlen=status_smoothing_window)
last_written_status = None
last_pred_time = 0
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ لم يتم الحصول على صورة من الكاميرا.")
        break

    processed_frame = preprocess_image(frame)
    frame_resized = cv2.resize(processed_frame, frame_size)
    frame_norm = frame_resized.astype('float32') / 255.0
    frame_buffer.append(frame_norm)

    # تنبؤ كل نصف ثانية
    current_time = time.time()
    if len(frame_buffer) == sequence_length and (current_time - last_pred_time) >= pred_interval:
        sequence = np.expand_dims(np.array(frame_buffer), axis=0)
        pred = model.predict(sequence, verbose=0)[0][0]
        status = "Awake" if pred > 0.5 else "Drowsy"
        status_history.append(status)
        smoothed_status = max(set(status_history), key=status_history.count)
        last_pred_time = current_time

        if smoothed_status != last_written_status:
            with open(status_path, 'w') as f:
                f.write(smoothed_status)
            last_written_status = smoothed_status

    # عرض الحالة
    if status_history:
        display_status = max(set(status_history), key=status_history.count)
        color = (0, 0, 255) if display_status == "Drowsy" else (0, 255, 0)
        cv2.putText(frame, display_status, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # FPS
    frame_count += 1
    if frame_count >= 10:
        end_time = time.time()
        fps = frame_count / (end_time - start_time)
        start_time = end_time
        frame_count = 0
        cv2.putText(frame, f"FPS: {fps:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # عرض الفيديو
    cv2.imshow('Driver Drowsiness Detection (Enhanced)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
