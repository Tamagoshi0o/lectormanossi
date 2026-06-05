"""
Sign Language Recognition - Vocal Recognition System
Sistema de Reconocimiento de Lenguaje de Señas - Reconocimiento de Vocales

Author / Autor: David Tamayo Villegas
Subject / Asignatura: Programación Estructurada
Evaluator / Docente: Robinson Damián Gómez Sánchez
Date / Fecha: 17/04/2026

Requirements covered / Requerimientos cubiertos:
  G5 - String handling & f-strings / Tratamiento de Cadenas y f-strings
  G6 - Dynamic storage (Lists), sorting & .pop() / Almacenamiento dinámico, ordenamiento
  G7 - Complex records with nested dictionaries & .update() / Diccionarios anidados
  BIL - Bilingual interface / Interfaz bilingüe
"""

import os
import time
import urllib.request
from datetime import datetime

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ─────────────────────────────────────────────
#  MODEL SETUP / CONFIGURACIÓN DEL MODELO
# ─────────────────────────────────────────────
MODEL_URL = "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# ─────────────────────────────────────────────
#  G5 – STRING HANDLING / TRATAMIENTO DE CADENAS
# ─────────────────────────────────────────────
def get_session_info() -> dict:
    print("\n" + "=" * 55)
    print(" SIGN LANGUAGE RECOGNITION / RECONOCIMIENTO DE SEÑAS ")
    print("=" * 55)

    # G5: .strip() and .upper() implementation
    raw_name = input("EN: Enter Engineer Name / ES: Nombre del Ingeniero: ")
    user_name = raw_name.strip().upper()

    raw_session = input("EN: Session ID / ES: ID de Sesión: ")
    session_id = raw_session.strip()

    while not session_id.isdigit():
        print("EN: Invalid ID. / ES: ID no válido.")
        session_id = input("EN: Session ID / ES: ID de Sesión: ").strip()

    return {"user_name": user_name, "session_id": session_id}


# ─────────────────────────────────────────────
#  G7 – NESTED DICTIONARIES / DICCIONARIOS ANIDADOS
# ─────────────────────────────────────────────
def create_session_db(user_name: str, session_id: str) -> dict:
    # G7: Transition from simple variables to robust nested dictionaries
    return {
        "user_info": {  # Nested Dict 1
            "name": user_name,
            "session_id": session_id,
            "start_time": datetime.now().strftime("%H:%M:%S"),
        },
        "stats": {  # Nested Dict 2
            "total": 0, "A": 0, "E": 0, "I": 0, "O": 0, "U": 0,
        },
        "sign_log": [],  # G6: Dynamic List for multiple records
    }


# ─────────────────────────────────────────────
#  G6 – LIST OPERATIONS / OPERACIONES DE LISTA
# ─────────────────────────────────────────────
def log_sign(session_db: dict, vowel: str):
    # G6: .append() implementation to save data in memory
    session_db["sign_log"].append(vowel)
    session_db["stats"]["total"] += 1
    session_db["stats"][vowel] += 1


def remove_last_sign(session_db: dict):
    # G6: .pop() implementation to dynamically remove errors
    if session_db["sign_log"]:
        removed = session_db["sign_log"].pop()
        session_db["stats"]["total"] -= 1
        session_db["stats"][removed] = max(0, session_db["stats"][removed] - 1)
        print(f"EN: Removed: {removed} / ES: Eliminado: {removed}")


# ─────────────────────────────────────────────
#  G7 – UPDATE FUNCTION / FUNCIÓN ACTUALIZAR
# ─────────────────────────────────────────────
def update_user(session_db: dict):
    new_name = input("EN: Enter Correct Name / ES: Ingrese Nombre Correcto: ").strip().upper()
    # G7: .update() implementation to modify complex record fields by key
    session_db["user_info"].update({"name": new_name})
    print("EN: Record updated. / ES: Registro actualizado.")


# ─────────────────────────────────────────────
#  G5 – FORMATTED REPORT / REPORTE (f-strings)
# ─────────────────────────────────────────────
def print_report(session_db: dict):
    # G6: Sorting values before displaying final results
    session_db["sign_log"].sort()

    info = session_db["user_info"]
    stats = session_db["stats"]

    print("\n" + "=" * 55)
    print(f"{' FINAL REPORT / REPORTE FINAL ':=^55}")
    print(f"  {'ENGINEER / INGENIERO':<20}: {info['name']}")
    print(f"  {'SESSION / SESIÓN':<20}: {info['session_id']}")
    print("-" * 55)
    # G5: Width modifiers for perfectly aligned tables
    print(f"  {'VOWEL / VOCAL':<20} | {'COUNT / CONTEO':<20}")
    print("-" * 55)
    for v in ["A", "E", "I", "O", "U"]:
        print(f"  {v:<20} | {stats[v]:<20}")
    print("-" * 55)
    print(f"  {'TOTAL':<20} | {stats['total']:<20}")
    print("=" * 55 + "\n")


# ─────────────────────────────────────────────
#  DETECTION LOGIC / LÓGICA DE DETECCIÓN
# ─────────────────────────────────────────────
def detect_vowel(landmarks, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    index_up = pts[8][1] < pts[5][1]
    middle_up = pts[12][1] < pts[9][1]
    ring_up = pts[16][1] < pts[13][1]
    pinky_up = pts[20][1] < pts[17][1]

    if not index_up and not middle_up and not ring_up and not pinky_up: return "A"
    if index_up and middle_up and ring_up and pinky_up: return "E"
    if index_up and not middle_up and not ring_up and not pinky_up: return "I"
    if index_up and middle_up and not ring_up and not pinky_up: return "U"
    return "O"


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("EN: Downloading model... / ES: Descargando modelo...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


# ─────────────────────────────────────────────
#  MAIN EXECUTION / EJECUCIÓN PRINCIPAL
# ─────────────────────────────────────────────
def main():
    session_info = get_session_info()
    db = create_session_db(session_info["user_name"], session_info["session_id"])
    ensure_model()

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO, num_hands=1
    )

    last_log = 0
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        time.sleep(2.0)  # Prevents "Empty Frame" initialization crash

        print("\nEN: Camera running. ESC: Quit | R: Remove last sign | U: Update Name")
        print("ES: Cámara activa. ESC: Salir | R: Borrar última seña | U: Actualizar Nombre")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: continue  # Keeps searching frames if hardware lags

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, ts)

            if result.hand_landmarks:
                h, w, _ = frame.shape
                # Simple landmark visualization
                for lm in result.hand_landmarks[0]:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)

                vowel = detect_vowel(result.hand_landmarks[0], w, h)
                cv2.putText(frame, f"Letter: {vowel}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                if (ts - last_log) > 1500:  # Log a record into dynamic list every 1.5s
                    log_sign(db, vowel)
                    last_log = ts

            # HUD Displaying Total Dynamic Count from nested dict
            total_count = db["stats"]["total"]
            cv2.putText(frame, f"Detections: {total_count}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

            cv2.imshow("FESC - Sign Language MVP", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break  # ESC
            elif key == ord('r') or key == ord('R'):
                remove_last_sign(db)
            elif key == ord('u') or key == ord('U'):
                update_user(db)

        cap.release()
        cv2.destroyAllWindows()

    # Print formatted output report with f-string width modifiers
    print_report(db)


if __name__ == "__main__":
    main()