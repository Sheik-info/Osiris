import subprocess
from django.shortcuts import render
from django.http import HttpResponse
from am1.main import *
import json
import random
from django.http import JsonResponse
from django.conf import settings
import os
    

auto1 = AutoEncodeur()

img = auto1.decompression()

print(img)

def index(request):
    return render(request, "index.html")

def ajouter_personne(chaine, fichier='personnes.txt'):
    champs = ["Nom", "Prenom", "Yeux", "Cheveux", "Sexe"]
    valeurs = chaine.strip().split(",")

    if len(valeurs) != len(champs):
        print("Erreur : la chaîne doit contenir exactement 5 éléments séparés par des virgules.")
        return

    data = {champ: [] for champ in champs}

    try:
        with open(fichier, "r") as f:
            lignes = f.readlines()
            for i in range(0, len(lignes), 2):
                champ = lignes[i].strip().strip(":")
                if i + 1 < len(lignes):
                    valeurs_ligne = lignes[i + 1].strip().strip(", \n").split(", ")
                    if champ in data:
                        data[champ] = valeurs_ligne
    except FileNotFoundError:
        pass

    for champ, valeur in zip(champs, valeurs):
        data[champ].append(valeur)

    with open(fichier, "w") as f:
        for champ in champs:
            f.write(f"{champ}:\n")
            if data[champ]:
                f.write(f" {', '.join(data[champ])},\n")
            else:
                f.write(" ,\n")

def ma_fonction(request):
    erreur = ""
    valide = False


    if request.method == 'POST':
        prenom = request.POST.get('prenom')
        nom = request.POST.get('nom')
        yeux = request.POST.get('yeux')
        cheveux = request.POST.get('cheveux')
        sexe = request.POST.get('sexe')

        if (yeux not in ['marron', 'bleu', 'vert'] or
                cheveux not in ['brun', 'chatain', 'blond', 'roux', 'blanc'] or
                sexe not in ['homme', 'femme']):
            erreur = "Veuillez entrer toutes les données correctement."
        else:
            valide = True


        return render(request, "fonct1.html", {'erreur': erreur})

    return render(request, "fonct1.html")


def jouer_akinator(request):
    json_file_path = os.path.join(settings.BASE_DIR, 'fichiers', 'questions.json')
    
    with open(json_file_path, 'r', encoding='utf-8') as file:
        questions_data = json.load(file)
    
    questions = questions_data['questions']

    question = random.choice(questions)

    if 'questions_posées' not in request.session:
        request.session['questions_posées'] = []

    if question['id'] not in request.session['questions_posées']:
        request.session['questions_posées'].append(question['id'])
    else:
        request.session['questions_posées'] = []
        question = random.choice(questions)
        request.session['questions_posées'].append(question['id'])

    return render(request, 'jouer_akinator.html', {'question': question})

def enregistrer_reponse(request):
    if request.method == 'POST':
        question_id = request.POST.get('question_id')
        reponse = request.POST.get('reponse')

        return JsonResponse({'status': 'réponse enregistrée'})

