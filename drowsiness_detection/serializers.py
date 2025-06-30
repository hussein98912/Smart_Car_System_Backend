from rest_framework import serializers

class ImageDataSerializer(serializers.Serializer):
    image = serializers.CharField()
