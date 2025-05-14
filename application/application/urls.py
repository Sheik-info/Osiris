from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),        # pour accéder à /
    path('am1/', views.index, name='am1'),       # pour accéder à /am1/
    path('executer/', views.ma_fonction, name="executer_fonction"),
    path('jouer_akinator/', views.jouer_akinator, name='jouer_akinator'),
    path('enregistrer_reponse/', views.enregistrer_reponse, name='enregistrer_reponse'),

]