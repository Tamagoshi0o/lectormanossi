"""
Detector de lenguaje de señas usando MediaPipe y OpenCV.
Detecta el abecedario básico y señas como Hola, Gracias, Sí, No.
"""

import os

import cv2
import numpy as np

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat


class HandDetector:
    """Clase para detectar y rastrear los 21 puntos clave de la mano."""

    def __init__(self, max_hands=2, detection_conf=0.7, tracking_conf=0.5,
                 model_path="hand_landmarker.task"):
        if not os.path.isfile(model_path):
            raise RuntimeError(
                "No se encontro el modelo 'hand_landmarker.task'. Descargalo de "
                "https://developers.google.com/mediapipe/solutions/vision/hand_landmarker "
                "y guardalo junto a este script."
            )

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_conf,
            min_hand_presence_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def find_hands(self, frame):
        """Procesa el frame y retorna los landmarks detectados."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        results = self.landmarker.detect(mp_image)
        hands_data = []

        if results.hand_landmarks:
            height, width = frame.shape[:2]
            for hand_landmarks in results.hand_landmarks:
                landmarks = []
                for lm in hand_landmarks:
                    landmarks.append((lm.x, lm.y, lm.z))
                    x_px = int(lm.x * width)
                    y_px = int(lm.y * height)
                    cv2.circle(frame, (x_px, y_px), 4, (0, 255, 0), -1)
                hands_data.append(landmarks)

        return frame, hands_data

    def close(self):
        self.landmarker.close()


class SignClassifier:
    """
    Clasificador de señas basado en geometría de landmarks.
    Para mayor precisión, reemplazar con un modelo Keras/TensorFlow.
    """

    def __init__(self):
        # Índices de puntas de cada dedo: pulgar, índice, medio, anular, meñique
        self.tip_ids = [4, 8, 12, 16, 20]
        # Índices de articulaciones PIP
        self.pip_ids = [3, 6, 10, 14, 18]

    def _fingers_up(self, landmarks):
        """Retorna lista de 5 booleanos indicando si cada dedo está extendido."""
        fingers = []
        # Pulgar: comparar x (para mano derecha)
        if landmarks[self.tip_ids[0]][0] < landmarks[self.tip_ids[0] - 2][0]:
            fingers.append(True)
        else:
            fingers.append(False)
        # Resto de dedos: comparar y (menor y = más arriba)
        for i in range(1, 5):
            fingers.append(landmarks[self.tip_ids[i]][1] < landmarks[self.pip_ids[i]][1])
        return fingers

    def _distance(self, p1, p2):
        """Distancia euclidiana entre dos puntos."""
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def classify(self, landmarks):
        """
        Clasifica la seña según la posición de los dedos.
        Retorna el texto reconocido.
        """
        if not landmarks:
            return ""

        fingers = self._fingers_up(landmarks)
        total_up = sum(fingers)

        # --- Señas específicas ---

        # "Hola" = todos los dedos extendidos, mano abierta
        if all(fingers):
            return "Hola"

        # "Sí" = puño cerrado (ningún dedo extendido)
        if total_up == 0:
            return "Si (Puno)"

        # "No" = índice y medio extendidos juntos, resto cerrado
        if fingers == [False, True, True, False, False]:
            # Verificar que índice y medio estén juntos
            dist = self._distance(landmarks[8], landmarks[12])
            if dist < 0.05:
                return "No"
            return "V / Victoria"

        # "Gracias" = palma abierta tocando mentón (simplificado: 4 dedos arriba sin pulgar)
        if fingers == [False, True, True, True, True]:
            return "Gracias"

        # --- Letras del abecedario (ASL simplificado) ---

        # A = puño con pulgar al lado
        if fingers == [True, False, False, False, False]:
            return "A"

        # B = 4 dedos arriba, pulgar cerrado
        if fingers == [False, True, True, True, True]:
            return "B / Gracias"

        # C = mano curvada (todos semi-extendidos)
        # L = pulgar e índice extendidos
        if fingers == [True, True, False, False, False]:
            return "L"

        # I = solo meñique
        if fingers == [False, False, False, False, True]:
            return "I"

        # Y = pulgar y meñique
        if fingers == [True, False, False, False, True]:
            return "Y"

        # D = solo índice
        if fingers == [False, True, False, False, False]:
            return "D / Uno"

        # W = índice, medio, anular
        if fingers == [False, True, True, True, False]:
            return "W / Tres"

        return f"Dedos: {total_up}"


def main():
    """Bucle principal de captura y detección."""
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    classifier = SignClassifier()

    print("Presiona 'q' para salir")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Espejo para naturalidad
            frame = cv2.flip(frame, 1)

            # Detectar manos
            frame, hands = detector.find_hands(frame)

            # Clasificar cada mano detectada
            for i, landmarks in enumerate(hands):
                sign = classifier.classify(landmarks)
                if sign:
                    # Mostrar texto en pantalla
                    y_pos = 50 + i * 40
                    cv2.putText(frame, f"Sena: {sign}", (10, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # Instrucciones
            cv2.putText(frame, "Presiona 'q' para salir", (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("Detector de Lenguaje de Senas", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

