import cv2
import mediapipe as mp
import streamlit as st
import pandas as pd
import numpy as np

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Definisci i colori e lo spessore personalizzati per i punti e le linee
landmark_drawing_spec = mpDraw.DrawingSpec(color=(255, 102, 0), thickness=2, circle_radius=2)  # Arancione
connection_drawing_spec = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2)  # Verde

while True:
    success, image = cap.read()
    if success:
        # ribalta l'immagine verticalmente
        image = cv2.flip(image, 1)
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(imageRGB)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # checking whether a hand is detected
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:  # working with each hand
            for id, lm in enumerate(handLms.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                if id == 8:
                    cv2.circle(image, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

            # Modifica i colori e lo spessore dei punti e delle linee qui
            mpDraw.draw_landmarks(
                image,
                handLms,
                mpHands.HAND_CONNECTIONS,
                landmark_drawing_spec=landmark_drawing_spec,
                connection_drawing_spec=connection_drawing_spec
            )

    cv2.imshow("Output", image)
    cv2.waitKey(1)
