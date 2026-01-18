from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    path('', views.landing, name='landing'),
    path('login/', auth_views.LoginView.as_view(template_name='predictor/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='landing'), name='logout'),
    path('predict/', views.predict, name='predict'),
    path('about/', views.about, name='about'),
]
