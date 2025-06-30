from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import os
import cv2
import numpy as np
import collections
import tensorflow as tf
import threading

# Load model once globally
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'DDD', 'my_model.keras')
status_path = os.path.join(BASE_DIR, 'DDD', 'status')
model = tf.keras.models.load_model(model_path)

# Drowsiness Detection Function
def run_drowsiness_detection():
    sequence_length = 5
    frame_size = (128, 128)
    frame_buffer = collections.deque(maxlen=sequence_length)
    last_written_status = None

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, frame_size)
        frame_filtered = cv2.GaussianBlur(frame_resized, (3, 3), 0)
        frame_filtered = cv2.convertScaleAbs(frame_filtered, alpha=1.2, beta=20)
        frame_norm = frame_filtered.astype('float32') / 255.0

        frame_buffer.append(frame_norm)

        if len(frame_buffer) == sequence_length:
            sequence = np.expand_dims(np.array(frame_buffer), axis=0)
            pred = model.predict(sequence)
            status_text = "Drowsy" if pred[0][0] > 0.5 else "Awake"

            if status_text != last_written_status:
                with open(status_path, 'w') as f:
                    f.write(status_text)
                last_written_status = status_text

            cv2.putText(frame, status_text, (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Driver Drowsiness Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# GET current status
class DrowsinessStatusAPIView(APIView):
        def get(self, request):
            try:
                if os.path.exists(status_path):
                    with open(status_path, 'r') as f:
                        status_text = f.read().strip()
                    return Response({"status": status_text}, status=status.HTTP_200_OK)
                else:
                    return Response({"error": "Status file not found."}, status=status.HTTP_404_NOT_FOUND)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# POST to start detection
class StartDrowsinessDetectionAPIView(APIView):
    def post(self, request):
        try:
            # Run detection in a separate thread to avoid blocking
            threading.Thread(target=run_drowsiness_detection, daemon=True).start()
            return Response({"message": "Drowsiness detection started."}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
