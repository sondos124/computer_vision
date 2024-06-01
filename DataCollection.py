# Importation des bibliothèques nécessaires
import cv2  # Bibliothèque OpenCV pour les tâches de vision par ordinateur
from cvzone.HandTrackingModule import HandDetector  # Module de suivi des mains
import numpy as np  # NumPy pour les opérations numériques
import math  # Bibliothèque mathématique pour les fonctions mathématiques
import time  # Module time pour les opérations liées au temps

# Initialisation de la capture vidéo à partir de la caméra par défaut (0)
cap = cv2.VideoCapture(0)

# Création d'un objet HandDetector avec un nombre maximum de mains fixé à 1
detector = HandDetector(maxHands=1)

# Configuration des paramètres pour le traitement de l'image
offset = 20  # Décalage pour recadrer la région de la main
imgSize = 300  # Taille de l'image de sortie
folder = "data/hello"  # Dossier pour enregistrer les images capturées
counter = 0  # Compteur pour suivre le nombre d'images enregistrées

# Boucle principale pour le traitement continu des images
while(True):
    # Lecture d'une trame de la caméra
    success, img = cap.read()

    # Détection des mains dans l'image en utilisant HandDetector
    hands, img = detector.findHands(img)

    # Vérification si des mains sont détectées
    if hands:
        # Extraction des informations sur la première main détectée
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Création d'un canevas blanc de la taille spécifiée
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Recadrage de la région de la main à partir de l'image
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape

        # Calcul du rapport hauteur/largeur de la région de la main
        aspectRatio = h / w

        # Redimensionnement et positionnement de l'image de la main recadrée sur le canevas blanc
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Affichage de l'image de la main recadrée et du canevas blanc
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # Affichage de l'image originale avec la détection des mains
    cv2.imshow("Image", img)

    # Attente d'une pression de touche et enregistrement de l'image si 's' est pressé
    key = cv2.waitKey(1)
    if key == ord('s'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
