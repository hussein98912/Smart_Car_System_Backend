import requests
from tqdm import tqdm
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os
import sys
import torchvision.transforms as T

# Add MiDaS repo path (must contain midas/dpt_depth.py etc.)
sys.path.append("midas")  # تأكد أن المسار صحيح وحرف صغير "midas"

# Import MiDaS modules
from midas.dpt_depth import DPTDepthModel

def download_midas_model(url, output_path):
    if os.path.exists(output_path):
        print(f"✅ Model already downloaded: {output_path}")
        return

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, 'wb') as f, tqdm(
        desc=f"Downloading {output_path}",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(1024):
            f.write(data)
            bar.update(len(data))

# Step 1: Download MiDaS model
midas_url = "https://github.com/isl-org/MiDaS/releases/download/v3/dpt_hybrid_384.pt"
midas_model_path = "dpt_hybrid_384.pt"
download_midas_model(midas_url, midas_model_path)

# Step 2: Load MiDaS model manually
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas = DPTDepthModel(
    path=midas_model_path,
    backbone="vitb_rn50_384",
    non_negative=True,
)
midas.to(device)
midas.eval()

# Step 3: Define transform using torchvision.transforms
dpt_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet means
        std=[0.229, 0.224, 0.225]    # ImageNet stds
    ),
])

# Step 4: Load YOLOv8 model
yolo_model = YOLO(r'front_camera_detection\models\best (1).pt')  # غيّر اسم الموديل لو عندك غيره

# Step 5: Video paths
video_path = r"media\0310.mp4"
output_path = r"media\output_with_distances.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ Error opening video file {video_path}")
    exit()

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

    frame_count += 1
    small_frame = cv2.resize(frame, (384, 384))
    img_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Step 6: Depth prediction
    input_tensor = dpt_transform(img_rgb).to(device)
    with torch.no_grad():
        prediction = midas(input_tensor.unsqueeze(0))  # أضف batch dim
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

    depth_map = (prediction - prediction.min()) / (prediction.max() - prediction.min())

    # Step 7: YOLO detection
    results = yolo_model(img_rgb)
    detections = results[0]

    scale_x = width / 384
    scale_y = height / 384

    for box, score, cls in zip(detections.boxes.xyxy, detections.boxes.conf, detections.boxes.cls):
        x1, y1, x2, y2 = map(int, box.cpu().numpy())
        if y2 > depth_map.shape[0] or x2 > depth_map.shape[1]:
            continue

        cropped_depth = depth_map[y1:y2, x1:x2]
        if cropped_depth.size == 0:
            continue

        estimated_distance = (1.0 - np.median(cropped_depth)) * 10

        label = yolo_model.names[int(cls.cpu().numpy())]
        text = f"{label}: {estimated_distance:.2f} m"

        x1_orig, y1_orig = int(x1 * scale_x), int(y1 * scale_y)
        x2_orig, y2_orig = int(x2 * scale_x), int(y2 * scale_y)

        cv2.rectangle(frame, (x1_orig, y1_orig), (x2_orig, y2_orig), (0, 255, 0), 2)
        cv2.putText(frame, text, (x1_orig, y1_orig - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)

# لا عرض للفيديو
print(f"✅ Finished processing {frame_count} frames.")
print(f"📁 Output saved at {output_path}")

cap.release()
out.release()
cv2.destroyAllWindows()
