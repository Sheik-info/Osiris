import numpy as np
from PIL import Image
from am1.Autoencodeur import AutoEncodeur
from am1.ReseauProfond import deep_neural_network




def load_and_prepare_image(image_path, target_size=(64, 64)):
    """
    Charge une image, la convertit en niveaux de gris, la redimensionne et la normalise.
    """
    try:
        origine = Image.open(image_path)
        img = origine.convert('L')  # Convertir en niveaux de gris
        img = img.resize(target_size)  # Redimensionner à la taille cible

        # Convertir l'image en tableau NumPy
        img_array = np.array(img, dtype=np.float32)  # Forme : (64, 64)

        # Normaliser et reshaper
        img_array = img_array / img_array.max()  # Normaliser entre 0 et 1
        X_reshape = img_array.reshape(-1, 1)  # Forme : (4096, 1) pour une seule image

        return X_reshape
    except Exception as e:
        print(f"Erreur lors du chargement de l'image {image_path} : {e}")
        return None
    
 #c'est oas un vrai GAN un vrai GAN le génère tout seul l'image il ne le prend pas aléatoirement   
def GAN():
    auto1 = AutoEncodeur()

    # Charger les données
    X = auto1.decompression()


    # Appeler deep_neural_network
    deep_neural_network(X, [1])


if __name__ == "__main__":
    #GAN()
    #mets le lien d'une image ici
    auto1 = AutoEncodeur()
    auto1.decompression()