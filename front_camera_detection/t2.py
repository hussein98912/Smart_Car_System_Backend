import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import sys

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLO models
signal_model = YOLO(r"front_camera_detection/models/best.pt")
vehicle_model = YOLO(r"front_camera_detection/models/best (1).pt")

# Load MiDaS depth model
sys.path.append("midas")
from midas.dpt_depth import DPTDepthModel
midas = DPTDepthModel("dpt_hybrid_384.pt", backbone="vitb_rn50_384", non_negative=True).to(device).eval()

# Load road damage model
road_damage_model_path = r"C:\Users\slman\Desktop\smart_car_backend\front_camera_detection\models\resnet_model.pt"
road_damage_model = torch.load(road_damage_model_path, map_location=device)
road_damage_model.eval()

# Define transforms
midas_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

road_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Road labels
road_labels = ['Good', 'Poor', 'Satisfactory', 'Very poor']

def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_num = 0
    close_vehicles = []
    detected_signals = []
    road_status_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        output_frame = frame.copy()

        if frame_num % 4 == 0:
            # Depth estimation
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (384, 384))
            input_tensor = midas_transform(img_resized).unsqueeze(0).to(device)

            with torch.no_grad():
                depth_prediction = midas(input_tensor)
                depth_prediction = torch.nn.functional.interpolate(
                    depth_prediction.unsqueeze(1),
                    size=(img_rgb.shape[0], img_rgb.shape[1]),
                    mode="bicubic",
                    align_corners=False
                ).squeeze().cpu().numpy()

            depth_map = (depth_prediction - depth_prediction.min()) / (depth_prediction.max() - depth_prediction.min() + 1e-6)

            # Vehicle detection + distance estimation
            vehicle_results = vehicle_model(img_rgb)[0]
            for box, cls, conf in zip(vehicle_results.boxes.xyxy, vehicle_results.boxes.cls, vehicle_results.boxes.conf):
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                cropped_depth = depth_map[y1:y2, x1:x2]
                if cropped_depth.size == 0:
                    continue

                median_depth = np.median(cropped_depth)
                median_depth = max(median_depth, 1e-6)
                scale_factor = 1.4
                estimated_distance = 1 / median_depth * scale_factor
                estimated_distance = np.clip(estimated_distance, 0.3, 20)

                label = vehicle_model.names[int(cls)]
                close_vehicles.append({
                    "frame": frame_num,
                    "vehicle_type": label,
                    "distance_m": estimated_distance
                })

                text = f"{label}: {estimated_distance:.2f} m"
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(output_frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Traffic signals detection (no overlay on output video)
            signal_results = signal_model(frame)[0]
            for box, cls in zip(signal_results.boxes.xyxy, signal_results.boxes.cls):
                label = signal_model.names[int(cls)]
                detected_signals.append({
                    "frame": frame_num,
                    "signal_type": label,
                    "bbox": [int(v.item()) for v in box]
                })

            # Road damage detection
            road_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            road_img_tensor = road_transform(Image.fromarray(road_img)).unsqueeze(0).to(device)
            with torch.no_grad():
                road_pred = road_damage_model(road_img_tensor)
                road_label_idx = torch.argmax(road_pred, dim=1).item()
            road_label = road_labels[road_label_idx]

            road_status_list.append({
                "frame": frame_num,
                "road_status": road_label
            })

            # Draw big road status label with black outline
            road_label_text = f"ROAD STATUS: {road_label.upper()}"
            cv2.putText(output_frame, road_label_text, (30, 70),
                        cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 0, 0), 6, cv2.LINE_AA)  # black outline
            cv2.putText(output_frame, road_label_text, (30, 70),
                        cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 255, 255), 2, cv2.LINE_AA)  # yellow text

        out.write(output_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Processing complete. Output saved to: {output_path}")
    return close_vehicles, detected_signals, road_status_list


if __name__ == "__main__":
    input_video_path = "media/05_20230626.mp4"
    output_video_path = "media/output_processed.mp4"
    close_vehicles, detected_signals, road_status_list = process_video(input_video_path, output_video_path)
