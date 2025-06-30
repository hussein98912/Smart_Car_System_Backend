import numpy as np
import tensorflow as tf

# تحميل الموديل المدرب مسبقًا
model = tf.keras.models.load_model(r"C:\Users\slman\Desktop\smart_car_backend\drive_class\dc\model.h5")

# قائمة الأصناف (يجب أن تطابق ترتيب الطبقة الأخيرة للموديل)
event_labels = ["sharp_turn", "normal_driving", "idle", "hard_brake"]

def predict_driving_pattern(data):
    input_array = np.array([[ 
        data["gps_speed"],
        data["speed_diff"],
        data["angle_change"],
        data["angular_velocity"],
        data["angular_acceleration"],
        data["angle"],
        data["angular_acceleration_smoothed"],
        data["distance_diff"],
        data["jerk"]
    ]])
    
    # التنبؤ بالاحتمالات
    prediction_probs = model.predict(input_array)

    # أخذ التصنيف الأعلى احتمالاً
    predicted_class_index = np.argmax(prediction_probs)
    predicted_event = event_labels[predicted_class_index]

    return {
        "predicted_event": predicted_event,
        "confidence": float(prediction_probs[0][predicted_class_index])  # يمكن إضافتها لعرض الدقة
    }
