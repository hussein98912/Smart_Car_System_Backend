from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('drowsiness_detection.urls')),  
    path('api/', include('drive_class.urls')),
    path('api/front/', include('front_camera_detection.urls')),
    path('', include('smart_assistance.urls')),  

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)