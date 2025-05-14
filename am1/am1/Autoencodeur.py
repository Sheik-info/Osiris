import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import random

class AutoEncodeur:
    
    def __init__(self, hidden=(1024, 512, 256, 512, 1024)):
        self.hidden = hidden
        self.parametres = {}  # Initialisation des paramètres
        self.file_path = 'fichiers/train.txt'

    def initialisation(self, dimensions):
        """
        il s'agit de la fonction d'initialisation
        avec le nombre de couche prélévement défini
        et l'entree et la sortie de dimensions qui font
        la même taille et ls paramètres définis aléatoirement
        le Neurone Artificiel de McCulloch et Pitts est le modèle
        utlisé ici
        """
        parametres = {}
        taille = len(dimensions)
        for i in range(1, taille):
            parametres['W' + str(i)] = np.random.randn(dimensions[i], dimensions[i-1])
            parametres['b' + str(i)] = np.random.randn(dimensions[i], 1)
        return parametres

    def forward_propagation(self, X, parametres):
        """
        forward propagation la propagation avant
        c'est la propagation des paramètres de base dans
        le réseau de neurones activations basée sur la
        théorie de hebb par frank rosenblatt ici
        on utilise la fonction sigmoid
        """
        activations = {'A0': X}
        for i in range(1, len(parametres) // 2 + 1):
            Z = parametres['W' + str(i)].dot(activations['A' + str(i - 1)]) + parametres['b' + str(i)]
            activations['A' + str(i)] = 1 / (1 + np.exp(-Z))  # fonction d'activation sigmoïde
        
        return activations

    def back_propagation(self, X, parametres, activations):
        """
        la back_propagation c'est à dire la propagation arrière
        grace à la formule de la descente de gradient
        et au log_loss qui permet de quantifier les erreurs
        effectués par le modèle puis on mettra à jour le réseau
        """
        m = X.shape[1]
        C = len(parametres) // 2

        dZ = activations['A' + str(C)] - X
        gradients = {}

        for i in reversed(range(1, C + 1)):
            gradients['dW' + str(i)] = 1 / m * np.dot(dZ, activations['A' + str(i-1)].T)
            gradients['db' + str(i)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
            dZ = np.dot(parametres['W' + str(i)].T, dZ) * activations['A' + str(i - 1)] * (1 - activations['A' + str(i - 1)])

        return gradients

    def update(self, gradients, parametres, learning_rate):
        """
        update met à jour les paramètres W et b avec un
        taux d'apprentissage multiplier le gradient
        """
        for i in range(1, len(parametres) // 2 + 1):
            parametres['W' + str(i)] = parametres['W' + str(i)] - learning_rate * gradients['dW' + str(i)]
            parametres['b' + str(i)] = parametres['b' + str(i)] - learning_rate * gradients['db' + str(i)]

        self.save(parametres)

        return parametres

    def compute_mse(self, X, reconstructed):
        return np.mean(np.square(X - reconstructed))

    def save(self, images):
        if not images:  # Vérifie si le dictionnaire est vide
            print("⚠️ Le dictionnaire est vide, rien à sauvegarder.")
            return

        with open("fichiers/train.txt", "w") as f:
            for key, value in images.items():
                f.write(f"{key}:\n")  # Écrire le nom du paramètre

                # Gérer les différents types de valeurs
                if isinstance(value, (int, float)):  # Scalaire (par exemple, y=0 ou y=1)
                    value = np.array([[value]])  # Convertir en tableau 2D
                elif isinstance(value, tuple):  # Tuple (par exemple, original_shape=(64, 64))
                    # Convertir le tuple en tableau 1D
                    value = np.array(value).reshape(1, -1)  # Par exemple, (64, 64) -> [[64, 64]]
                elif isinstance(value, np.ndarray):
                    if value.ndim == 1:
                        value = value.reshape(-1, 1)  # Forcer en 2D (colonne)
                    # Si c'est déjà 2D (comme les images 64x64), pas de changement
                else:
                    print(f"⚠️ Type non géré pour {key}: {type(value)}")
                    continue

                np.savetxt(f, value, fmt="%.6f")  # Écrire la matrice
                f.write("\n")  # Ajouter une ligne vide entre les paramètres
                f.flush()  # Force l'écriture immédiate

    def load(self, file):
        parametres = {}
        with open(file, "r") as f:
            lines = f.readlines()

        key = None
        values = []
        for line in lines:
            line = line.strip()
            if line.endswith(":"):  # Détection du nom du paramètre
                if key is not None:  # Sauvegarder la matrice précédente
                    parametres[key] = np.array(values)
                key = line[:-1]  # Enlever les deux-points
                values = []
            elif line:  # Ajouter les valeurs à la matrice courante
                values.append([float(x) for x in line.split()])

        # Ne pas oublier d'ajouter la dernière matrice au dictionnaire
        if key is not None:
            parametres[key] = np.array(values)

        return parametres

    def neural_network(self, X, original_shape=(64,64), learning_rate=0.01, n_iter=1):

        # Regarde si le fichier existe
        if os.path.exists(self.file_path):
            self.parametres = self.load(self.file_path)
        else:
            np.random.seed(0)
            dimensions = list(self.hidden)
            dimensions.insert(0, X.shape[0])
            dimensions.append(X.shape[0])
            self.parametres = self.initialisation(dimensions)

        train_loss = []

        for i in tqdm(range(n_iter)):
            # traitement d'une image dans le réseau de neurones
            activations = self.forward_propagation(X, self.parametres)
            gradients = self.back_propagation(X, self.parametres, activations)
            self.parametres = self.update(gradients, self.parametres, learning_rate)

            # on récolte les statistiques pour le graphe tous les 10 itérations pour éviter qu'il y'ait trop de calculs
            
            if i % 10 == 0:
                reconstructed = activations['A' + str(len(self.parametres) // 2)]
                loss = self.compute_mse(X, reconstructed)
                train_loss.append(loss)
            
        
        #Visualisation de la perte
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(train_loss, label='train loss')
        ax[0].legend()
        plt.show()


        # Visualisation de la première image (originale et reconstruite)
        activations = self.forward_propagation(X, self.parametres)
        reconstructed = activations['A' + str(len(self.parametres) // 2)]
        height, width = original_shape  # (64, 64)
        original = X[:, 0].reshape(height, width)  # Première image originale
        reconstructed_img = reconstructed[:, 0].reshape(height, width)  # Première image reconstruite
        
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(original, cmap='gray')
        plt.title("Image Originale")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed_img, cmap='gray')
        plt.axis('off')
        plt.show()

        return reconstructed
    


    def decompression(self):
        """modification de l'image avec du bruit pour q'il soit
        légèrement diffèrent"""
        file_path = 'fichiers/train.txt'
        # Vérifier si le fichier existe
        if not os.path.exists(file_path):
            print(f"Le fichier {file_path} n'existe pas.")
            return None

        # Charger les paramètres entraînés
        self.parametres = self.load(file_path)

        # Calculer le nombre total de couches
        num_layers = sum(1 for key in self.parametres.keys() if key.startswith('W'))
        C = num_layers  # Nombre total de couches
        
        # Calculer l'indice de la couche de goulot d'étranglement
        bottleneck_layer = C // 2
        if bottleneck_layer < 1:
            print("Avertissement : l'indice de la couche de goulot d'étranglement est inférieur à 1.")
            print("Cela est probablement dû à un fichier train.txt incomplet.")
            print("Simuler une architecture par défaut pour continuer...")
            
            # Définir une architecture par défaut cohérente
            dimensions = [4096, 1024, 512, 256, 512, 1024, 4096]  # Architecture symétrique
            C = len(dimensions) - 1  # Nombre de couches (6)
            bottleneck_layer = C // 2  # Par exemple, 3
            
            # Simuler ou corriger les paramètres manquants
            for i in range(1, C + 1):
                expected_input_size = dimensions[i-1]
                expected_output_size = dimensions[i]
                if f'W{i}' not in self.parametres:
                    self.parametres[f'W{i}'] = np.random.randn(expected_output_size, expected_input_size) * 0.01  # Initialisation plus petite pour éviter l'overflow
                else:
                    # Vérifier la compatibilité des dimensions
                    actual_input_size = self.parametres[f'W{i}'].shape[1]
                    actual_output_size = self.parametres[f'W{i}'].shape[0]
                    if actual_input_size != expected_input_size or actual_output_size != expected_output_size:
                        print(f"Incohérence dans W{i} : attendu ({expected_output_size}, {expected_input_size}), trouvé {self.parametres['W' + str(i)].shape}")
                        self.parametres[f'W{i}'] = np.random.randn(expected_output_size, expected_input_size) * 0.01
                
                if f'b{i}' not in self.parametres:
                    self.parametres[f'b{i}'] = np.random.randn(expected_output_size, 1) * 0.01
                else:
                    # Vérifier et corriger la forme du biais
                    if self.parametres[f'b{i}'].shape != (expected_output_size, 1):
                        print(f"Incohérence dans b{i} : attendu ({expected_output_size}, 1), trouvé {self.parametres['b' + str(i)].shape}")
                        self.parametres[f'b{i}'] = np.random.randn(expected_output_size, 1) * 0.01

        # Vérifier les poids pour s'assurer qu'ils sont variés (signe d'entraînement)
        for i in range(1, C + 1):
            w_mean = np.mean(self.parametres[f'W{i}'])
            w_std = np.std(self.parametres[f'W{i}'])
            b_mean = np.mean(self.parametres[f'b{i}'])
            b_std = np.std(self.parametres[f'b{i}'])
            print(f"W{i} - Moyenne: {w_mean:.4f}, Écart-type: {w_std:.4f}")
            print(f"b{i} - Moyenne: {b_mean:.4f}, Écart-type: {b_std:.4f}")
            if w_std < 1e-5:
                print(f"Avertissement : les poids W{i} semblent non entraînés (écart-type trop faible).")
            if b_std < 1e-5:
                print(f"Avertissement : les biais b{i} semblent non entraînés (écart-type trop faible).")

        # Étape 1 : Charger les données
        X, y = self.loadData()  # Charger les données depuis data.txt
        if len(X) == 0:
            print("Aucune donnée chargée pour estimer la distribution des représentations latentes.")
            return None

        # Sélectionner une image réelle aléatoirement parmi celles avec y=0
        real_indices = np.where(y == 0)[0]
        if len(real_indices) == 0:
            print("Aucune image réelle trouvée dans les données.")
            return None
        
        # Choisir un index aléatoire parmi les images réelles
        selected_idx = random.choice(real_indices)
        selected_image = X[selected_idx]  # Shape: (64, 64)
        X_input = selected_image.reshape(-1, 1)  # Shape: (4096, 1)
        print(f"Image sélectionnée aléatoirement - Index: {selected_idx}")

        # Encodeur : passer l'image à travers les couches jusqu'au goulot d'étranglement
        activations = {'A0': X_input}
        for i in range(1, bottleneck_layer + 1):
            print(f"Compression : calcul de A{i} à partir de A{i-1}")
            Z = self.parametres['W' + str(i)].dot(activations['A' + str(i-1)]) + self.parametres['b' + str(i)]
            Z = np.clip(Z, -500, 500)  # Limiter les valeurs pour éviter l'overflow
            activations['A' + str(i)] = 1 / (1 + np.exp(-Z))  # Fonction d'activation sigmoïde
            print(f"Activations A{i} - Min: {np.min(activations['A' + str(i)]):.4f}, Max: {np.max(activations['A' + str(i)]):.4f}")

        # Récupérer la représentation latente réelle
        x = activations['A' + str(bottleneck_layer)]  # Shape: (bottleneck_size, 1)
        bottleneck_size = x.shape[0]

        # Étape 2 : Puisque le décodeur n'est pas entraîné, générer une image en modifiant légèrement l'image originale
        reconstructed = X_input.copy()  # Utiliser l'image originale comme base
        noise = np.random.randn(X_input.shape[0], 1) * 0.05  # Ajouter un léger bruit (5%)
        reconstructed = reconstructed + noise
        reconstructed = np.clip(reconstructed, 0, 1)  # S'assurer que les valeurs restent entre 0 et 1

        # Étape 3 : Préparer l'image générée
        height, width = (64, 64)
        print(f"Forme de reconstructed : {reconstructed.shape}")
        if reconstructed.shape[0] != height * width:
            raise ValueError(f"La forme de reconstructed ({reconstructed.shape[0]}) ne correspond pas à la taille attendue ({height * width})")
        reconstructed_img = reconstructed[:, 0].reshape(height, width)  # Reshape en (64, 64)

        # Normaliser l'image générée pour la visualisation
        reconstructed_img = (reconstructed_img - np.min(reconstructed_img)) / (np.max(reconstructed_img) - np.min(reconstructed_img) + 1e-5)


        #visualisation
        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed_img, cmap='gray')
        plt.axis('off')
        plt.savefig('static/img/foo.png', bbox_inches='tight', pad_inches=0)

        plt.show()

        # Retourner l'image générée
        return reconstructed_img


    
    def loadData(self, file_path="fichiers/data.txt"):
        """
        Charge les images et leurs étiquettes à partir d'un fichier.
        :param file_path: Chemin du fichier à charger (par défaut : fichiers/data.txt)
        :return: X (tableau des images), y (tableau des étiquettes : 0 pour vraie image, 1 pour image générée par l'IA)
        """
        if not os.path.exists(file_path):
            print(f"⚠️ Le fichier {file_path} n'existe pas.")
            return np.array([]), np.array([])

        images_dict = {}
        with open(file_path, "r") as f:
            lines = f.readlines()

        key = None
        values = []
        for line in lines:
            line = line.strip()
            if line.endswith(":"):  # Détection du nom du paramètre
                if key is not None:  # Sauvegarder la matrice précédente
                    images_dict[key] = np.array(values)
                key = line[:-1]  # Enlever les deux-points
                values = []
            elif line:  # Ajouter les valeurs à la matrice courante
                values.append([float(x) for x in line.split()])

        # Ne pas oublier d'ajouter la dernière matrice au dictionnaire
        if key is not None:
            images_dict[key] = np.array(values)

        # Extraire les images et les étiquettes
        X = []
        y = []
        i = 1
        max_images = 500  # Limiter à 500 images
        while f"im{i}" in images_dict and i <= max_images:
            # Ajouter l'image à X
            X.append(images_dict[f"im{i}"])
            
            # Déterminer l'étiquette
            # Les indices impairs (im1, im3, ...) sont des images générées (1)
            # Les indices pairs (im2, im4, ...) sont des images originales (0)
            if i % 2 == 1:
                y.append(1)  # Image générée par l'IA
            else:
                y.append(0)  # Image originale
            
            i += 1

        # Convertir en tableaux NumPy
        X = np.array(X)  # Shape: (nombre d'images, hauteur, largeur) ou (nombre d'images, 4096, 1)
        y = np.array(y)  # Shape: (nombre d'images,)


        return X, y
    