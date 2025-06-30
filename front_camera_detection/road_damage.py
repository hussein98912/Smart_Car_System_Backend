import cv2
import torch
from torchvision import transforms

# تحميل النموذج من PyTorch
model_path = r"C:\Users\slman\Desktop\smart_car_backend\front_camera_detection\models\resnet_model.pt"
model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
model.eval()

# معالجة الفيديو
def process_video():
    input_path = r'media/processed2_output.mp4'
    output_path = r'media/processed3_output.mp4'

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # حسب ما يحتاجه نموذجك
        transforms.ToTensor()
    ])

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess(frame).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)

        # افترض أن النموذج يعطي class فقط (أضف logic حسب نوع النموذج)
        predicted_class = torch.argmax(outputs, dim=1).item()

        label = f"Class: {predicted_class}"
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
