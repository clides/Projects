from django.db import models

# Create your models here.
class ImageUpload(models.Model):
    image = models.ImageField(upload_to='uploads/')  # Store images in an 'uploads' directory

    def __str__(self):
        return self.image.name