# 🌌 Projet AM1 - Osiris

### 🧠 Domaines
**Image Processing / Deep Learning (IA) / Interaction**

### 🛠️ Techniques et langages
**Python / HTML / CSS**

### 👨‍🏫 Responsable du sujet
**Alexandre Meyer**

---

##  Objectif du projet

Le projet **AM1 - Osiris** est un projet en **deep learning** axé sur la **reconnaissance et la génération d'images**.

Nous avons décidé d'intégrer ces concepts dans un **site web interactif** utilisant :
- Un **Autoencodeur** capable de régénérer une image qu’on lui donne.
- Un **réseau de neurones profond** entraîné pour **reconnaître si une image est générée par une IA** ou non.

Ces deux modèles sont mis en opposition dans un système proche des **GANs (Generative Adversarial Networks)** :  
 l’un tente de tromper l’autre.

L'objectif global du projet est d'**explorer les possibilités offertes par les réseaux de neurones dans le traitement d'images**, avec un focus sur **la génération d’images** à travers des mini-jeux ludiques.

---

##  Concept du site web

Le site est basé sur le thème des **stars de cinéma** et comprend plusieurs mini-jeux, qui servent en réalité à **entraîner l’IA Osiris** :

- **Jeu de génération** : où l'utilisateur donne des caractéristiques (yeux, cheveux, sexe…) pour que l'IA génère un visage.
- **Akinator inversé** : l’IA tente de générer une star à partir d’indications de l’utilisateur, avec des limites dues à la quantité de données disponibles.

---

## 🗂️ Organisation des fichiers

   AM1-Osiris
   ├──  am1/                    # Scripts Python principaux

   │   ├── Autoencodeur.py        # Autoencodeur pour la génération d'images

   │   └── ReseauProfond.py       # Réseau pour la détection d'images IA

   ├──  documentation/          # Documents explicatifs et présentations

   ├──  fichiers/               # Fichiers utilitaires (entraînement, sauvegardes)

   ├──  static/                 # Ressources statiques

   │   ├──  css/                # Feuilles de style CSS

   │   └──  js/                 # Scripts JavaScript (optionnel)

   ├──  templates/              # Fichiers HTML pour l'interface web
   
   └── manage.py                # Script Django pour lancer le serveur



---

## 📦 Bibliothèques nécessaires

Voici les principales bibliothèques utilisées (à installer via `pip install <nom>` si besoin) :

- `h5py`
- `numpy`
- `opencv-python`
- `matplotlib`
- `tqdm`
- `scikit-learn`
- `pandas`
- `django`

---

## 🚀 Lancer le projet

1. Cloner le projet :
   ```bash
   git clone <lien-du-projet>
   cd am1-lifprojet
   pip install -r fichiers/requirements.txt
   python manage.py makemigrations
   python manage.py migrate


   ```
2. Lancer Django
   ```bash
   python manage.py runserver
   ```
<br />
## Equipe

```bash
Ba Cheikh
p2109987

Gayon Enzo 
p2102346

EL KARROUMI CHERAZADE
p2001916

```

