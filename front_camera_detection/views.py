from django.http import JsonResponse
from . import processor , processor2 

def api_close_vehicles(request):
    return JsonResponse({"close_vehicles": processor2.close_vehicles})

def api_detected_signals(request):
    return JsonResponse({"detected_signals": processor2.detected_signals})

def api_road_status(request):
    return JsonResponse({"road_status": processor2.road_status_list})

def start_video_processing(request):
    print("Processing started") 
    video_path = "media/The Green and Red Traffic Lights in Action.mp4"
    processor2.process_video(video_path)
    return JsonResponse({
        "message": "proccessing started ..",
    })
