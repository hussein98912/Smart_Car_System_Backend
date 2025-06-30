from django.db import models

class DetectedVehicle(models.Model):
    vehicle_type = models.CharField(max_length=50)
    distance = models.FloatField()
    frame_number = models.IntegerField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.vehicle_type} - {self.distance:.2f}m"
