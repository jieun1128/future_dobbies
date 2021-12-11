from rest_framework import serializers

class SearchBodySerializer(serializers.Serializer):
    imageURL = serializers.ListField(help_text="이미지 검색 후 나온 URL들", child=serializers.CharField())
    text = serializers.ListField(help_text="OCR 결과 문장들", child=serializers.CharField())
