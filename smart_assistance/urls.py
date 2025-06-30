from django.urls import path
from .views import start_wake_word_listener
from .views import IntentFromFileAPIView
from .views import wake_word_status_view


urlpatterns = [
    path('assistant/api/start-wake-word/', start_wake_word_listener, name='wake_word_api'),
    path('assistant/api/get-intent/', IntentFromFileAPIView.as_view(), name='get_intent'),
    path('assistant/api/wake-word-status/', wake_word_status_view, name='wake_word_status'),
]

