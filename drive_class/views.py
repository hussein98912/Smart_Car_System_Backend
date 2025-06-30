from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import numpy as np
import tensorflow as tf


model = tf.keras.models.load_model(r"C:\Users\slman\Desktop\smart_car_backend\drive_class\dc\model.h5")


event_labels = ["sharp_turn", "normal_driving", "idle", "hard_brake"]

@csrf_exempt
def predict_view(request):
    if request.method == "POST":
        try:
          
            body = json.loads(request.body)

           
            if not isinstance(body, list):
                return JsonResponse({"error": "Expected a list of data points"}, status=400)

        
            sequence = []
            for data in body:
                features = [
                    data["gps_speed"],
                    data["speed_diff"],
                    data["angle_change"],
                    data["angular_velocity"],
                    data["angular_acceleration"],
                    data["angle"],
                    data["angular_acceleration_smoothed"],
                    data["distance_diff"],
                    data["jerk"]
                ]
                sequence.append(features)

            if len(sequence) != 10:
                return JsonResponse({"error": "Expected exactly 10 data points"}, status=400)

            input_array = np.array(sequence).reshape(1, 10, 9)

            # التنبؤ دفعة واحدة
            prediction_probs = model.predict(input_array)

            results = []
            for i in range(prediction_probs.shape[0]): 
                for j in range(prediction_probs.shape[1]):  
                   
                    pass

            predicted_class_index = np.argmax(prediction_probs[0])
            predicted_event = event_labels[predicted_class_index]
            confidence = float(prediction_probs[0][predicted_class_index])

            return JsonResponse({
                "predicted_event": predicted_event,
                "confidence": confidence
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"message": "Only POST method is allowed"}, status=405)
