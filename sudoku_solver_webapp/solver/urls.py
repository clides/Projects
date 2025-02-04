from django.urls import path
from . import views

urlpatterns = [
    path('solve/', views.solve_sudoku_image, name='solve_sudoku_image'),
    path('', views.upload_image, name='upload_image'),
]