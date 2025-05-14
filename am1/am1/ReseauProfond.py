import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
import seaborn as sns  # Pour la matrice de confusion
from tqdm import tqdm
import os



def initialisation(dimensions):
    parametres = {}
    taille = len(dimensions)
    for i in range(1, taille):
        parametres['W' + str(i)] = np.random.randn(dimensions[i], dimensions[i-1])
        parametres['b' + str(i)] = np.random.randn(dimensions[i], 1)
    return parametres



def forward_propagation(X, parametres):
    # Initialiser avec A0 (couche d'entrée)
    activations = {'A0': X}
    for i in range(1, len(parametres) // 2 + 1):
        Z = parametres['W' + str(i)].dot(activations['A' + str(i - 1)]) + parametres['b' + str(i)]
        activations['A' + str(i)] = 1 / (1 + np.exp(-Z))  # fonction d'activation sigmoïde
                
    return activations


def back_propagation(X, y, parametres, activations):
    m = y.shape[1]
    C = len(parametres) // 2

    dZ = activations['A' + str(C)] - y
    gradients = {}

    for i in reversed(range(1, C + 1)):
        gradients['dW' + str(i)] = 1 / m * np.dot(dZ, activations['A' + str(i-1)].T)
        gradients['db' + str(i)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        if i > 1:
            dZ = np.dot(parametres['W' + str(i)].T, dZ) * activations['A' + str(i-1)] * (1 - activations['A' + str(i-1)])
    return gradients



def update(gradients, parametres, learning_rate):
    C = len(parametres) // 2
    for i in range(1, C + 1):
        parametres['W' + str(i)] = parametres['W' + str(i)] - learning_rate * gradients['dW' + str(i)]
        parametres['b' + str(i)] = parametres['b' + str(i)] - learning_rate * gradients['db' + str(i)]
    
    save(parametres)

    return parametres



def predict(X, parametres):
    activations = forward_propagation(X, parametres)
    C = len(parametres) // 2
    A_final = activations['A' + str(C)]
    return (A_final >= 0.5).astype(int)  # Convertir en 0 ou 1


def deep_neural_network(X, y, hidden=(32, 32, 32), learning_rate=0.01, n_iter=1):
    # le fichier de sauvegarde
    file_path = '../fichiers/savaProfond.txt'

    # Regarde si le fichier existe
    if os.path.exists(file_path):
        parametres = load(file_path)
    else:
        # Initialisation des paramètres
        np.random.seed(0)

        dimensions = list(hidden)
        dimensions.insert(0, X.shape[0])  # X.shape[0] doit être 4096 (nombre de caractéristiques)
        dimensions.append(y.shape[0])     # y.shape[0] doit être 1 (classification binaire)
        parametres = initialisation(dimensions)

    train_loss = []
    train_acc = []

    # Gradient descent
    for i in tqdm(range(n_iter)):
        activations = forward_propagation(X, parametres)
        gradients = back_propagation(X, y, parametres, activations)
        parametres = update(gradients, parametres, learning_rate)

        if i % 10 == 0:
            A_final = activations['A' + str(len(parametres) // 2)]
            train_loss.append(log_loss(y.flatten(), A_final.flatten()))
            y_pred = predict(X, parametres)
            current_accuracy = accuracy_score(y.flatten(), y_pred.flatten())
            train_acc.append(current_accuracy)


    # Visualisation de la perte et de la précision
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(train_loss, label='train loss')
    ax[0].set_title('Loss')
    ax[0].legend()
    ax[1].plot(train_acc, label='train acc')
    ax[1].set_title('Accuracy')
    ax[1].legend()
    plt.show()

    # Prédictions sur les données d'entraînement
    y_pred_train = predict(X, parametres)

    # Matrice de confusion pour les données d'entraînement
    cm_train = confusion_matrix(y.flatten(), y_pred_train.flatten())
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Matrice de confusion (Entraînement)")
    plt.xlabel("Prédiction")
    plt.ylabel("Vraie étiquette")
    plt.show()

    return parametres

def save(neurones):
    if not neurones:  # Vérifie si le dictionnaire est vide
        print("⚠️ Le dictionnaire est vide, rien à sauvegarder.")
        return

    with open("../fichiers/savaProfond.txt", "w") as f:
        for key, value in neurones.items():
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

def load(file):
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
