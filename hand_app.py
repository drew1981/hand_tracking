import base64
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from io import BytesIO

st.title("Hand Tracking")

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

landmark_drawing_spec = mpDraw.DrawingSpec(color=(255, 102, 0), thickness=1, circle_radius=1)  # Arancione
connection_drawing_spec = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2)  # Verde

def process_image(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
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

st.markdown("""
    <style>
        video {
            max-width: 100%;
            height: auto;
        }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <script>
        async function startCamera() {
            const video = document.getElementById('webcam-video');
            const constraints = {
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: "user"
                }
            };
            video.srcObject = await navigator.mediaDevices.getUserMedia(constraints);
            await video.play();
        }
        startCamera();
    </script>
    <video id="webcam-video" autoplay playsinline></video>
    """, unsafe_allow_html=True)

img_data = st.text_input("Incolla qui i dati dell'immagine in base64:")

if img_data:
    header, data = img_data.split(',')
    decoded_img = base64.b64decode(data)
    img = Image.open(BytesIO(decoded_img))
    
    processed_img = process_image(img)
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    st.image(processed_img, caption='Processed Image', use_column_width=True)

