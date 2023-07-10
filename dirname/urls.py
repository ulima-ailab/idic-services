"""
URL configuration for dirname project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from .mainView import firebase_collection, get_emotions, start_message_generation

urlpatterns = [
    path('admin/', admin.site.urls),
    path('firebase-collection/', firebase_collection, name='firebase_collection'),
    path('get-emotions/', get_emotions, name='get_emotions'),
    path('start-message-generation/', start_message_generation, name='start_message_generation'),
]
