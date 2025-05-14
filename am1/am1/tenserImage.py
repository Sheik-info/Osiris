import tenserflow as tf
import cv2
import numpy as np



def testIA(uneImage):

    #charger le modèle
    model = tf.keras.applications.MobileNetV2(weights='imagenet')

    #charger l'image
    image = cv2.imread(uneImage)

    #prétraiter l'image
    resized = cv2.resize(image, (224, 224))
    resized = tf.keras.preprocessing.image.img_to_array(resized)
    resized = tf.keras.applications.mobilenet_v2.preprocess_input(resized)

    #pediction sur l'image
    predictions =  model.predict(np.array([resized]))
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)
    max_val = 0
    label_max = ''

    #afficher les résultats
    print(max(decoded_predictions))
    for _, label, score in decoded_predictions[0]:
        max_val, label_max = score.max, label
    print(f"ceci est un {label_max}")
   #for _, label, score in decoded_predictions[0]:
       #print(f"ceci est peut être {label} : probabilité {score}")
