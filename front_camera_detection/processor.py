import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import sys
import os

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

close_vehicles = []
detected_signals = []
road_status_list = []

def process_video(video_path):
    global close_vehicles, detected_signals, road_status_list
    close_vehicles.clear()
    detected_signals.clear()
    road_status_list.clear()

    cap = cv2.VideoCapture(video_path)
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        if frame_num % 4 == 0:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (384, 384))
            input_tensor = midas_transform(img_resized).unsqueeze(0).to(device)

            with torch.no_grad():
                prediction = midas(input_tensor)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=(img_rgb.shape[0], img_rgb.shape[1]),
                    mode="bicubic",
                    align_corners=False
                ).squeeze().cpu().numpy()

            depth_map = (prediction - prediction.min()) / (prediction.max() - prediction.min())

            vehicle_results = vehicle_model(img_rgb)[0]
            for box, cls in zip(vehicle_results.boxes.xyxy, vehicle_results.boxes.cls):
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                cropped_depth = depth_map[y1:y2, x1:x2]
                if cropped_depth.size == 0:
                    continue
                distance = (1.0 - np.median(cropped_depth)) * 10
                label = vehicle_model.names[int(cls)]

                print("Detected vehicle:", label, "at distance:", distance)

                if distance < 8.0:
                    close_vehicles.append({
                        "frame": frame_num,
                        "vehicle_type": label,
                        "distance_m": distance
                    })

            signal_results = signal_model(frame)[0]
            for box, cls in zip(signal_results.boxes.xyxy, signal_results.boxes.cls):
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                label = signal_model.names[int(cls)]
                detected_signals.append({
                    "frame": frame_num,
                    "signal_type": label,
                    "bbox": [x1, y1, x2, y2]
                })

            road_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            road_img_tensor = road_transform(Image.fromarray(road_img)).unsqueeze(0).to(device)

            with torch.no_grad():
                road_pred = road_damage_model(road_img_tensor)
                road_label = torch.argmax(road_pred, dim=1).item()

            road_status_list.append({
                "frame": frame_num,
                "road_status": road_label
            })

    cap.release()
