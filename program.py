# Importation des bibliothèques nécessaires
import cv2  # Librairie OpenCV pour les tâches de vision par ordinateur
from cvzone.HandTrackingModule import HandDetector  # Module de suivi des mains
from cvzone.ClassificationModule import Classifier  # Module de classification d'images
import numpy as np  # NumPy pour les opérations numériques
import math  # Bibliothèque mathématique pour les fonctions mathématiques

# Initialisation de la capture vidéo à partir de la caméra par défaut (0)
cap = cv2.VideoCapture(0)

# Création des objets HandDetector et Classifier
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Configuration des paramètres pour le traitement de l'image
offset = 20  # Décalage pour recadrer la région de la main
imgSize = 300  # Taille de l'image de sortie
folder = "Data/E"  # Dossier pour enregistrer les images capturées
counter = 0  # Compteur pour suivre le nombre d'images enregistrées

# Labels pour la classification
labels = ["hello"]

# Boucle principale pour le traitement continu des images
while True:
    # Lecture d'une trame de la caméra
    success, img = cap.read()
    imgOutput = img.copy()

    # Détection des mains dans l'image en utilisant HandDetector
    hands, img = detector.findHands(img)

    # Vérification de la présence de mains détectées
    if hands:
        # Extraction des informations sur la première main détectée
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Création d'un canevas blanc de la taille spécifiée
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Recadrage de la région de la main à partir de l'image
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        # Redimensionnement et positionnement de l'image de la main recadrée sur le canevas blanc
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # Ajout d'une boîte autour de la région de la main et affichage du label prédit
        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)
        print(labels[index])

        # Affichage de l'image de la main recadrée et du canevas blanc
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # Affichage de l'image originale avec la détection des mains et la classification
    cv2.imshow("Image", imgOutput)

    # Attente d'une pression de touche (1 milliseconde)
    cv2.waitKey(1)
