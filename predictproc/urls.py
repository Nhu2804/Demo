from django.urls import path
from . import views

app_name = "predictproc"

urlpatterns = [
    path("", views.procedure_page, name="procedure_page"),
    path("api/", views.procedure_api, name="procedure_api"),
]
