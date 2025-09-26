from django.urls import path
from .views import query_rag

urlpatterns = [
    path("query/", query_rag, name="query_rag"),
]
