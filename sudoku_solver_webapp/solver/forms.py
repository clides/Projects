from django import forms
from .models import ImageUpload

class ImageUploadForm(forms.Form):
    image = forms.ImageField()