import cv2
from ultralytics import YOLO
import os

model = YOLO(r"C:\Users\slman\Desktop\smart_car_backend\front_camera_detection\models\best.pt")

def process_video():
    input_path = r'media\processed_output.mp4'
    output_path = r'media\processed2_output.mp4'

    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        out.write(frame)
        frame_count += 1
        print(f"🟢 تمت معالجة الفريم {frame_count}")

    cap.release()
    out.release()

    print("✅ تم حفظ الفيديو بعد المعالجة")
    return output_path

if __name__ == '__main__':
    output_path = process_video()
    print(f"📁 الفيديو الناتج: {output_path}")
