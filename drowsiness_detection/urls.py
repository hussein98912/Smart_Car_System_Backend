from django.urls import path
from .views import DrowsinessStatusAPIView,StartDrowsinessDetectionAPIView

urlpatterns = [
    path('drowsiness/api/detect/', DrowsinessStatusAPIView.as_view(), name='drowsiness_api'),
    path('drowsiness/api/start/', StartDrowsinessDetectionAPIView.as_view(), name='start_drowsiness_detection'),
]
