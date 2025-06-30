from django.urls import path
from . import views

urlpatterns = [
    path('close_vehicles/', views.api_close_vehicles, name='close_vehicles'),
    path('detected_signals/', views.api_detected_signals, name='detected_signals'),
    path('road_status/', views.api_road_status, name='road_status'),
    path('start_processing/', views.start_video_processing, name='start_processing'),
]