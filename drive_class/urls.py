from django.urls import path
from .views import predict_view

urlpatterns = [
    path('predict-driving-pattern/', predict_view, name='predict-driving-pattern'),
]
