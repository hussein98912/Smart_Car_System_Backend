import os
import cv2
import torch
import numpy as np
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'midas'))

from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose

# إعداد الجهاز
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# مسارات الملفات
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(BASE_DIR, 'media', '0310.mp4')
output_dir = os.path.join(BASE_DIR, 'media', 'processed')
os.makedirs(output_dir, exist_ok=True)
output_video_path = os.path.join(output_dir, '0310_processed.mp4')

# تحميل موديل MiDaS
midas_model_path = os.path.join(BASE_DIR, 'front_camera_detection', 'models', 'dpt_hybrid_384.pt')

midas = DPTDepthModel(path=midas_model_path, backbone="vitb_rn50_384", non_negative=True).to(device).eval()

# تحميل موديل YOLO
from ultralytics import YOLO
yolo_model_path = os.path.join(BASE_DIR, 'front_camera_detection/models/best (1).pt')
yolo_model = YOLO(yolo_model_path)

# إنشاء التحويل
transform = Compose([
    Resize(
        384, 384,
        resize_target=None,
        keep_aspect_ratio=True,
        ensure_multiple_of=32,
        resize_method="minimal",
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    PrepareForNet(),
])

def process_video_with_depth_and_yolo(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Failed to open video")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    scale_x = width / 384
    scale_y = height / 384

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (384, 384))
        img_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # تقدير العمق
        input_tensor = transform(img_rgb).to(device).unsqueeze(0)
        with torch.no_grad():
            prediction = midas(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()

        depth_map = (prediction - prediction.min()) / (prediction.max() - prediction.min())

        # الكشف عن الأجسام
        results = yolo_model(img_rgb)
        detections = results[0]

        for box, score, cls in zip(detections.boxes.xyxy, detections.boxes.conf, detections.boxes.cls):
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            if y2 > depth_map.shape[0] or x2 > depth_map.shape[1]:
                continue

            cropped_depth = depth_map[y1:y2, x1:x2]
            estimated_distance = (1.0 - np.median(cropped_depth)) * 10

            label = yolo_model.names[int(cls.cpu().numpy())]
            text = f"{label}: {estimated_distance:.2f} m"

            x1_orig, y1_orig = int(x1 * scale_x), int(y1 * scale_y)
            x2_orig, y2_orig = int(x2 * scale_x), int(y2 * scale_y)

            cv2.rectangle(frame, (x1_orig, y1_orig), (x2_orig, y2_orig), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1_orig, y1_orig - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Video saved to {output_video_path}")
    return True

process_video_with_depth_and_yolo(input_video_path, output_video_path)
