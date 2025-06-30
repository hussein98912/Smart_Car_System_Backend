import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import sys

# تعاريف وأجهزة ونماذج كما في كودك (مخفف هنا)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

signal_model = YOLO(r"front_camera_detection/models/best.pt")
vehicle_model = YOLO(r"front_camera_detection/models/best (1).pt")

sys.path.append("midas")
from midas.dpt_depth import DPTDepthModel
midas = DPTDepthModel("dpt_hybrid_384.pt", backbone="vitb_rn50_384", non_negative=True).to(device).eval()

road_damage_model_path = r"C:\Users\slman\Desktop\smart_car_backend\front_camera_detection\models\resnet_model.pt"
road_damage_model = torch.load(road_damage_model_path, map_location=device)
road_damage_model.eval()

midas_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

road_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# المتغيرات لتخزين البيانات
close_vehicles = []
detected_signals = []
road_status_list = []

def process_video(video_path):
    global close_vehicles, detected_signals, road_status_list
    close_vehicles.clear()
    detected_signals.clear()
    road_status_list.clear()

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        # MiDaS depth
        small_frame = cv2.resize(frame, (384, 384))
        img_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        input_tensor = midas_transform(img_rgb).to(device)

        with torch.no_grad():
            prediction = midas(input_tensor.unsqueeze(0))
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(img_rgb.shape[0], img_rgb.shape[1]),
                mode="bicubic",
                align_corners=False
            ).squeeze().cpu().numpy()

        depth_map = (prediction - prediction.min()) / (prediction.max() - prediction.min())

        # كشف المركبات
        vehicle_results = vehicle_model(img_rgb)[0]
        for box, cls in zip(vehicle_results.boxes.xyxy, vehicle_results.boxes.cls):
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            cropped_depth = depth_map[y1:y2, x1:x2]
            if cropped_depth.size == 0:
                continue
            distance = (1.0 - np.median(cropped_depth)) * 10
            label = vehicle_model.names[int(cls)]

            # إذا المسافة أقل من متر واحد خزّن
            if distance < 1.0:
                close_vehicles.append({
                    "frame": frame_num,
                    "vehicle_type": label,
                    "distance_m": distance
                })

        # كشف إشارات المرور
        signal_results = signal_model(frame)[0]
        for box, cls in zip(signal_results.boxes.xyxy, signal_results.boxes.cls):
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            label = signal_model.names[int(cls)]
            detected_signals.append({
                "frame": frame_num,
                "signal_type": label,
                "bbox": [x1, y1, x2, y2]
            })

        # تصنيف حالة الطريق
        road_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        road_img_tensor = road_transform(Image.fromarray(road_img)).unsqueeze(0).to(device)
        with torch.no_grad():
            road_pred = road_damage_model(road_img_tensor)
            road_label = torch.argmax(road_pred, dim=1).item()

        labels_road = ["✅ الطريق سليم", "⚠️ حفر", "⚠️ تشققات", "⚠️ مطب"]
        status_text = labels_road[road_label] if road_label < len(labels_road) else "غير معروف"
        road_status_list.append({
            "frame": frame_num,
            "road_status": status_text
        })

    cap.release()
