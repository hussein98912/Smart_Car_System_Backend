import os
import cv2
import torch
import numpy as np
import sys

# إضافة مجلد midas إلى مسار الاستيراد
sys.path.append(os.path.join(os.path.dirname(__file__), "midas"))

# استيراد دالة التحميل الرسمية
from midas.model_loader import load_model

# إعداد الجهاز
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# مسار الموديل
model_path = "front_camera_detection/models/dpt_hybrid_384.pt"

# تحميل الموديل والتحويلات
model_type = "dpt_hybrid"
model, transform, net_w, net_h = load_model(device, model_path, model_type)

# تحميل صورة اختبار
img_path = "front_camera_detection/test_image.jpg"  # ضع صورة هنا للاختبار
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# تطبيق التحويلات
sample = transform({"image": img})
image_input = sample["image"].unsqueeze(0).to(device)

# تنفيذ التنبؤ بالعمق
with torch.no_grad():
    prediction = model.forward(image_input)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

depth = prediction.cpu().numpy()

# حفظ خريطة العمق كصورة
depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_MAGMA)
cv2.imwrite("front_camera_detection/depth_output.jpg", depth_colored)

print("✅ Depth map saved at front_camera_detection/depth_output.jpg")
