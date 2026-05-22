"""
Detector de lenguaje de senas / Sign language detector.
Proyecto simple con lista enlazada, pila, cola y archivo de texto.
Simple project with linked list, stack, queue and text file.
"""

import os
from collections import Counter, deque

import cv2
import numpy as np

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat


class Gesture:
    """Dato de una sena / Sign data."""

    def __init__(self, code, spanish, english):
        self.code = code.strip().upper()
        self.spanish = spanish.strip()
        self.english = english.strip()

    def show(self):
        """Texto para mostrar / Text to display."""
        return f"{self.code}: {self.spanish} / {self.english}"

    def to_file_line(self):
        """Linea para archivo / File line."""
        return f"{self.code}|{self.spanish}|{self.english}\n"


class Node:
    """Nodo basico / Basic node."""

    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    """Lista enlazada manual para Gesture / Manual linked list for Gesture."""

    def __init__(self):
        self.head = None

    def is_empty(self):
        """Revisa si esta vacia / Checks if empty."""
        return self.head is None

    def add_node(self, data):
        """Agrega al final / Adds at the end."""
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            return

        current = self.head
        while current.next is not None:
            current = current.next
        current.next = new_node

    def display_list(self):
        """Muestra la lista / Displays the list."""
        if self.head is None:
            print("No hay senas registradas. / No signs registered.")
            return

        current = self.head
        position = 1
        while current is not None:
            print(f"{position}. {current.data.show()}")
            current = current.next
            position += 1

    def find_node(self, code):
        """Busca por codigo / Searches by code."""
        code = code.strip().upper()
        current = self.head
        while current is not None:
            if current.data.code == code:
                return current.data
            current = current.next
        return None

    def update_node(self, code, spanish, english):
        """Actualiza una sena / Updates one sign."""
        code = code.strip().upper()
        current = self.head
        while current is not None:
            if current.data.code == code:
                if spanish.strip() != "":
                    current.data.spanish = spanish.strip()
                if english.strip() != "":
                    current.data.english = english.strip()
                return True
            current = current.next
        return False


class Stack:
    """Pila manual para historial / Manual stack for history."""

    def __init__(self):
        self.top = None

    def push(self, data):
        """Apila / Pushes."""
        new_node = Node(data)
        new_node.next = self.top
        self.top = new_node

    def pop(self):
        """Desapila / Pops."""
        if self.top is None:
            return None
        data = self.top.data
        self.top = self.top.next
        return data

    def display_stack(self, limit=10):
        """Muestra los ultimos movimientos / Shows latest moves."""
        if self.top is None:
            print("Historial vacio. / Empty history.")
            return

        current = self.top
        count = 1
        while current is not None and count <= limit:
            print(f"{count}. {current.data.show()}")
            current = current.next
            count += 1


class Queue:
    """Cola manual para procesamiento / Manual queue for processing."""

    def __init__(self):
        self.front = None
        self.rear = None

    def encolar(self, data):
        """Encola / Enqueues."""
        new_node = Node(data)
        if self.rear is None:
            self.front = new_node
            self.rear = new_node
            return
        self.rear.next = new_node
        self.rear = new_node

    def desencolar(self):
        """Desencola / Dequeues."""
        if self.front is None:
            return None
        data = self.front.data
        self.front = self.front.next
        if self.front is None:
            self.rear = None
        return data

    def display_queue(self):
        """Muestra la cola / Shows the queue."""
        if self.front is None:
            print("Cola vacia. / Empty queue.")
            return

        current = self.front
        position = 1
        while current is not None:
            print(f"{position}. {current.data.show()}")
            current = current.next
            position += 1


class HandDetector:
    """Detecta 21 puntos de la mano / Detects 21 hand landmarks."""

    def __init__(self, max_hands=2, detection_conf=0.7, tracking_conf=0.5,
                 model_path="hand_landmarker.task"):
        if not os.path.isfile(model_path):
            raise RuntimeError(
                "No se encontro el modelo 'hand_landmarker.task'. / "
                "The model 'hand_landmarker.task' was not found. "
                "Descargalo y guardalo junto a este script. / "
                "Download it and save it next to this script."
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
        """Procesa el frame / Processes the frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        results = self.landmarker.detect(mp_image)
        hands_data = ()

        if results.hand_landmarks:
            height, width = frame.shape[:2]
            for index, hand_landmarks in enumerate(results.hand_landmarks):
                landmarks = ()
                for lm in hand_landmarks:
                    landmarks = landmarks + ((lm.x, lm.y, lm.z),)
                    x_px = int(lm.x * width)
                    y_px = int(lm.y * height)
                    cv2.circle(frame, (x_px, y_px), 4, (0, 255, 0), -1)
                handedness = "Unknown"
                handedness_score = 0.0
                if results.handedness and len(results.handedness) > index and results.handedness[index]:
                    category = results.handedness[index][0]
                    handedness = category.category_name
                    handedness_score = category.score

                hands_data = hands_data + ({
                    "landmarks": landmarks,
                    "handedness": handedness,
                    "score": handedness_score,
                },)

        return frame, hands_data

    def close(self):
        """Cierra MediaPipe / Closes MediaPipe."""
        self.landmarker.close()


class SignClassifier:
    """Clasificador compacto para vocales / Compact vowel classifier."""

    def __init__(self):
        self.tip_ids = (4, 8, 12, 16, 20)
        self.pip_ids = (3, 6, 10, 14, 18)
        self.mcp_ids = (2, 5, 9, 13, 17)

    def _extract_hand(self, hand_data):
        """Acepta landmarks o un diccionario con metadatos / Accepts landmarks or metadata dict."""
        if isinstance(hand_data, dict):
            return hand_data.get("landmarks", ()), hand_data.get("handedness", "Unknown")
        return hand_data, "Unknown"

    def _palm_size(self, landmarks):
        """Tamano de palma para normalizar distancias / Palm size to normalize distances."""
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        palm_size = self._distance(wrist, middle_mcp)
        return max(palm_size, 1e-6)

    def _normalized_distance(self, landmarks, first_index, second_index):
        """Distancia normalizada por palma / Distance normalized by palm size."""
        return self._distance(landmarks[first_index], landmarks[second_index]) / self._palm_size(landmarks)

    def _clamp01(self, value):
        """Limita un valor al rango [0, 1] / Clamps a value to [0, 1]."""
        return max(0.0, min(1.0, float(value)))

    def _thumb_out_score(self, landmarks, handedness):
        """Apertura lateral del pulgar / Side opening of the thumb."""
        tip = landmarks[4]
        ip = landmarks[3]
        index_mcp = landmarks[5]
        thumb_index = self._normalized_distance(landmarks, 4, 5)

        if handedness == "Left":
            outward_delta = tip[0] - ip[0]
        elif handedness == "Right":
            outward_delta = ip[0] - tip[0]
        else:
            outward_delta = abs(tip[0] - ip[0])

        outward_score = self._clamp01((outward_delta - 0.01) / 0.05)
        spread_score = self._clamp01((thumb_index - 0.38) / 0.45)
        level_score = 1.0 - self._clamp01((abs(tip[1] - index_mcp[1]) - 0.08) / 0.45)
        return (0.45 * outward_score) + (0.4 * spread_score) + (0.15 * level_score)

    def _finger_open_score(self, landmarks, tip_index, pip_index, mcp_index):
        """Apertura de un dedo / Finger openness score."""
        wrist = landmarks[0]
        tip = landmarks[tip_index]
        pip = landmarks[pip_index]
        mcp = landmarks[mcp_index]
        palm_size = self._palm_size(landmarks)

        vertical_score = self._clamp01((pip[1] - tip[1] - 0.01) / 0.12)
        tip_to_wrist = self._distance(tip, wrist) / palm_size
        pip_to_wrist = self._distance(pip, wrist) / palm_size
        mcp_to_wrist = self._distance(mcp, wrist) / palm_size
        reach_score = self._clamp01((tip_to_wrist - pip_to_wrist - 0.05) / 0.45)
        mcp_score = self._clamp01((tip_to_wrist - mcp_to_wrist - 0.18) / 0.55)
        return (0.4 * vertical_score) + (0.35 * reach_score) + (0.25 * mcp_score)

    def _distance(self, p1, p2):
        """Distancia euclidiana / Euclidean distance."""
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _extract_features(self, landmarks, handedness):
        """Rasgos minimos y utiles para A/E/I/O/U / Minimal useful features for vowels."""
        index_open = self._finger_open_score(landmarks, 8, 6, 5)
        middle_open = self._finger_open_score(landmarks, 12, 10, 9)
        ring_open = self._finger_open_score(landmarks, 16, 14, 13)
        pinky_open = self._finger_open_score(landmarks, 20, 18, 17)
        thumb_out = self._thumb_out_score(landmarks, handedness)
        thumb_index = self._normalized_distance(landmarks, 4, 8)
        thumb_middle = self._normalized_distance(landmarks, 4, 12)
        index_middle = self._normalized_distance(landmarks, 8, 12)
        thumb_horizontal = abs(landmarks[4][0] - landmarks[5][0]) / self._palm_size(landmarks)
        thumb_vertical = abs(landmarks[4][1] - landmarks[5][1]) / self._palm_size(landmarks)

        return {
            "thumb_out": thumb_out,
            "index_open": index_open,
            "middle_open": middle_open,
            "ring_open": ring_open,
            "pinky_open": pinky_open,
            "index_closed": 1.0 - index_open,
            "middle_closed": 1.0 - middle_open,
            "ring_closed": 1.0 - ring_open,
            "pinky_closed": 1.0 - pinky_open,
            "thumb_index": thumb_index,
            "thumb_middle": thumb_middle,
            "index_middle": index_middle,
            "thumb_horizontal": thumb_horizontal,
            "thumb_vertical": thumb_vertical,
        }

    def classify_with_score(self, hand_data):
        """Clasifica vocales y devuelve confianza / Classifies vowels and returns confidence."""
        landmarks, handedness = self._extract_hand(hand_data)
        if not landmarks:
            return "", 0.0

        features = self._extract_features(landmarks, handedness)
        index_open = features["index_open"]
        middle_open = features["middle_open"]
        ring_open = features["ring_open"]
        pinky_open = features["pinky_open"]
        thumb_out = features["thumb_out"]
        thumb_index = features["thumb_index"]
        thumb_middle = features["thumb_middle"]
        index_middle = features["index_middle"]
        thumb_horizontal = features["thumb_horizontal"]
        thumb_vertical = features["thumb_vertical"]

        if (
            thumb_index > 0.68 and
            thumb_horizontal > 0.18 and
            thumb_vertical < 0.95 and
            index_open < 0.52 and
            middle_open < 0.52 and
            ring_open < 0.52 and
            pinky_open < 0.52
        ):
            return "A", max(thumb_out, min(1.0, thumb_index / 1.1))

        if (
            thumb_out < 0.50 and
            index_open < 0.42 and
            middle_open < 0.42 and
            ring_open < 0.42 and
            pinky_open < 0.42 and
            thumb_middle < 1.10
        ):
            return "E", 1.0 - max(index_open, middle_open, ring_open, pinky_open, thumb_out)

        if (
            pinky_open > 0.62 and
            index_open < 0.45 and
            middle_open < 0.45 and
            ring_open < 0.45
        ):
            return "I", pinky_open

        if (
            index_open > 0.58 and
            middle_open > 0.58 and
            ring_open < 0.45 and
            pinky_open < 0.45 and
            index_middle < 0.55
        ):
            return "U", min(index_open, middle_open)

        if (
            thumb_index < 0.52 and
            thumb_middle < 1.00 and
            index_open < 0.58 and
            middle_open < 0.58 and
            ring_open < 0.58 and
            pinky_open < 0.58
        ):
            return "O", 1.0 - thumb_index

        return "", 0.0

    def classify(self, hand_data):
        """Clasifica solo vocales / Classifies vowels only."""
        code, _ = self.classify_with_score(hand_data)
        return code


class DetectionSmoother:
    """Suaviza detecciones por frames / Smooths detections across frames."""

    def __init__(self, window_size=5, min_count=3, min_confidence=0.58):
        self.window_size = window_size
        self.min_count = min_count
        self.min_confidence = min_confidence
        self.history = {}

    def update(self, hand_index, candidate, confidence):
        """Devuelve una prediccion estable o vacia / Returns a stable prediction or empty."""
        if hand_index not in self.history:
            self.history[hand_index] = deque(maxlen=self.window_size)

        history = self.history[hand_index]
        history.append((candidate, confidence))
        valid_entries = [(code, score) for code, score in history if code and score >= self.min_confidence]
        valid_codes = [code for code, _ in valid_entries]
        if len(valid_codes) < self.min_count:
            return ""

        counts = Counter(valid_codes)
        code, count = counts.most_common(1)[0]
        average_confidence = sum(score for current_code, score in valid_entries if current_code == code) / count
        if count >= self.min_count and average_confidence >= self.min_confidence:
            return code
        return ""


def add_default_gestures(gestures):
    """Vocales iniciales / Initial vowels."""
    gestures.add_node(Gesture("A", "Vocal A", "Vowel A"))
    gestures.add_node(Gesture("E", "Vocal E", "Vowel E"))
    gestures.add_node(Gesture("I", "Vocal I", "Vowel I"))
    gestures.add_node(Gesture("O", "Vocal O", "Vowel O"))
    gestures.add_node(Gesture("U", "Vocal U", "Vowel U"))


def load_gestures(file_name):
    """Lee archivo a lista enlazada / Reads file into linked list."""
    gestures = LinkedList()
    try:
        with open(file_name, "r", encoding="utf-8") as file:
            for line in file:
                clean_line = line.strip()
                if clean_line != "":
                    parts = clean_line.split("|")
                    if len(parts) >= 3 and parts[0].strip().upper() in ("A", "E", "I", "O", "U"):
                        gestures.add_node(Gesture(parts[0], parts[1], parts[2]))
    except FileNotFoundError:
        print("Archivo no encontrado. / File not found. Se creara uno nuevo. / A new one will be created.")

    if gestures.is_empty():
        add_default_gestures(gestures)
        save_gestures(gestures, file_name)

    return gestures


def save_gestures(gestures, file_name):
    """Guarda lista enlazada en archivo / Saves linked list to file."""
    with open(file_name, "w", encoding="utf-8") as file:
        current = gestures.head
        while current is not None:
            file.write(current.data.to_file_line())
            current = current.next


def get_gesture(gestures, code):
    """Obtiene una sena o crea temporal / Gets a sign or creates temporary."""
    found = gestures.find_node(code.strip().upper())
    if found is not None:
        return found
    return Gesture(code, code, code)


def run_detector(gestures, history, processing_queue):
    """Ejecuta camara y detector / Runs camera and detector."""
    cap = cv2.VideoCapture(0)
    detector = None
    last_code = ""
    smoother = DetectionSmoother()
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

    if not cap.isOpened():
        print("No se pudo abrir la camara. / Could not open the camera.")
        return

    try:
        detector = HandDetector(model_path=model_path)
        classifier = SignClassifier()
        print("Presiona 'q' para salir. / Press 'q' to exit.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame, hands = detector.find_hands(frame)

            for i, hand_data in enumerate(hands):
                code, confidence = classifier.classify_with_score(hand_data)
                stable_code = smoother.update(i, code, confidence)
                if stable_code == "":
                    continue

                gesture = get_gesture(gestures, stable_code)
                y_pos = 45 + i * 70
                cv2.putText(frame, f"Sena/Sign: {gesture.code}", (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(frame, f"{gesture.spanish} / {gesture.english}", (10, y_pos + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Conf: {confidence:.2f}", (10, y_pos + 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 255, 160), 1)

                if stable_code != last_code:
                    history.push(gesture)
                    processing_queue.encolar(gesture)
                    last_code = stable_code

            cv2.putText(frame, "Presiona q / Press q", (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow("Detector de Senas / Sign Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except RuntimeError as error:
        print(f"Error / Error: {error}")
    finally:
        if detector is not None:
            detector.close()
        cap.release()
        cv2.destroyAllWindows()


def search_gesture(gestures):
    """Busca desde consola / Searches from console."""
    try:
        code = input("Codigo de la sena / Sign code: ").strip().upper()
    except EOFError:
        print("Entrada finalizada. / Input closed.")
        return

    gesture = gestures.find_node(code)
    if gesture is None:
        print("No encontrada. / Not found.")
    else:
        print(f"Encontrada / Found: {gesture.show()}")


def update_gesture(gestures):
    """Actualiza desde consola / Updates from console."""
    try:
        code = input("Codigo a actualizar / Code to update: ").strip().upper()
        spanish = input("Nuevo texto en espanol (Enter para dejar igual) / New Spanish text (Enter to keep): ")
        english = input("Nuevo texto en ingles (Enter para dejar igual) / New English text (Enter to keep): ")
    except EOFError:
        print("Entrada finalizada. / Input closed.")
        return

    if gestures.update_node(code, spanish, english):
        print("Sena actualizada. / Sign updated.")
    else:
        print("Sena no encontrada. / Sign not found.")


def process_queue(processing_queue):
    """Procesa una sena de la cola / Processes one sign from the queue."""
    gesture = processing_queue.desencolar()
    if gesture is None:
        print("No hay senas pendientes. / No pending signs.")
    else:
        print(f"Procesada / Processed: {gesture.show()}")


def show_menu():
    """Menu principal / Main menu."""
    print("\n--- Menu de Senas / Sign Menu ---")
    print("1. Iniciar detector / Start detector")
    print("2. Ver senas guardadas / Show saved signs")
    print("3. Buscar sena / Search sign")
    print("4. Actualizar sena / Update sign")
    print("5. Ver historial / Show history")
    print("6. Ver cola / Show queue")
    print("7. Procesar una sena / Process one sign")
    print("8. Guardar y salir / Save and exit")


def main():
    """Control top-down del programa / Top-down program control."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(base_path, "gestures.txt")
    gestures = load_gestures(file_name)
    history = Stack()
    processing_queue = Queue()
    running = True

    while running:
        show_menu()
        try:
            option = int(input("Opcion / Option: ").strip())
        except EOFError:
            save_gestures(gestures, file_name)
            print("Entrada finalizada. / Input closed. Datos guardados. / Data saved.")
            break
        except KeyboardInterrupt:
            save_gestures(gestures, file_name)
            print("\nInterrumpido. / Interrupted. Datos guardados. / Data saved.")
            break
        except ValueError:
            print("Opcion invalida. / Invalid option.")
            continue

        if option == 1:
            run_detector(gestures, history, processing_queue)
        elif option == 2:
            gestures.display_list()
        elif option == 3:
            search_gesture(gestures)
        elif option == 4:
            update_gesture(gestures)
            save_gestures(gestures, file_name)
        elif option == 5:
            history.display_stack()
        elif option == 6:
            processing_queue.display_queue()
        elif option == 7:
            process_queue(processing_queue)
        elif option == 8:
            save_gestures(gestures, file_name)
            print("Datos guardados. / Data saved. Adios / Goodbye.")
            running = False
        else:
            print("Opcion invalida. / Invalid option.")


if __name__ == "__main__":
    main()
