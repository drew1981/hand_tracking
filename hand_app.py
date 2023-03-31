import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.title("Hand Tracking")

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

landmark_drawing_spec = mpDraw.DrawingSpec(color=(255, 102, 0), thickness=1, circle_radius=1)  # Arancione
connection_drawing_spec = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2)  # Verde

def process_image(image):
    image = cv2.flip(image, 1)
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                if id == 8:
                    cv2.circle(image, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(
                image,
                handLms,
                mpHands.HAND_CONNECTIONS,
                landmark_drawing_spec=landmark_drawing_spec,
                connection_drawing_spec=connection_drawing_spec
            )

    return image

st.set_option('deprecation.showfileUploaderEncoding', False)
stframe = st.empty()

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Errore nella cattura del video.")
        break

    frame = process_image(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Aggiungi un contenitore per il frame del video
    container = st.container()
    with container:
        stframe.image(frame, channels='RGB', use_column_width=True)
