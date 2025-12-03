# predict_disease/urls.py
from django.urls import path
from . import views

app_name = 'predict_disease'  # Namespace để gọi url dễ hơn

urlpatterns = [
    path('', views.index, name='index'), # Trang chọn task
    path('heart/', views.predict_heart, name='predict_heart'),
    path('parkinson/', views.predict_parkinson, name='predict_parkinson'),
    # path('next-visit/', views.predict_next_visit, name='predict_next_visit'),
]