from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import threading
from .wake_words import start_listening
from smart_assistance import text_to_command
from rest_framework.views import APIView
from rest_framework.response import Response
import subprocess
from rest_framework import status
import os


@csrf_exempt
def start_wake_word_listener(request):
    if request.method == "POST":
        threading.Thread(target=start_listening).start()
        return JsonResponse({"status": "Wake word listener started."})
    else:
        return JsonResponse({"error": "Only POST method allowed."}, status=405)


class IntentFromFileAPIView(APIView):
    def get(self, request):
        intent_num, intent_name, confidence = text_to_command.predict_intent_from_file()

        # Set default intent_num to 500 if it's None or falsy
        if not intent_num:
            intent_num = 500

        if intent_num == 500 and (intent_name is None or confidence is None):
            # Optionally, return an error if no valid input was found
            return Response({"error": "No input text found or file is empty."}, status=status.HTTP_400_BAD_REQUEST)

        return Response({
            "intent_num": intent_num,
            "intent_name": intent_name,
            "confidence": confidence
        })

def wake_word_status_view(request):
    filepath = "wake_word_status.txt"
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            content = f.read().strip()
    else:
        content = None

    return JsonResponse({"wake_word_status": content})