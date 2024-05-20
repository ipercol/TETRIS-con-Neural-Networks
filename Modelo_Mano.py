import cv2
import mediapipe as mp
import math
import numpy as np


def main(cap):
    # Configuración de MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    gestures_labels = {
        0: "Unknown",
        1: "Piedra",
        2: "Papel",
        3: "Tijera",
    }

    # Inicialización de la clase Hands para la detección de manos
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    def is_closed_fist(threshold=0.35):
        # Calcular las distancias entre los dedos y la muñeca
        thumb_distance = math.sqrt((pulgar.x - munyeca.x)**2 + (pulgar.y - munyeca.y)**2)
        index_distance = math.sqrt((indice.x - munyeca.x)**2 + (indice.y - munyeca.y)**2)
        middle_distance = math.sqrt((corazon.x - munyeca.x)**2 + (corazon.y - munyeca.y)**2)
        ring_distance = math.sqrt((anular.x - munyeca.x)**2 + (anular.y - munyeca.y)**2)
        pinky_distance = math.sqrt((menyique.x - munyeca.x)**2 + (menyique.y - munyeca.y)**2)

        # Verificar si todos los dedos están cerrados y cerca de la muñeca
        if (thumb_distance < threshold and
            index_distance < threshold and
            middle_distance < threshold and
            ring_distance < threshold and
            pinky_distance < threshold):
            return True
        else:
            return False

    def is_victory(apertura=0.5, cerrado = 0.3):
        # Calcular las distancias entre los dedos
        thumb_distance = math.sqrt((pulgar.x - anular.x)**2 + (pulgar.y - anular.y)**2)
        index_distance = math.sqrt((indice.x - munyeca.x)**2 + (indice.y - munyeca.y)**2)
        middle_distance = math.sqrt((corazon.x - munyeca.x)**2 + (corazon.y - munyeca.y)**2)
        ring_distance = math.sqrt((anular.x - munyeca.x)**2 + (anular.y - munyeca.y)**2)
        pinky_distance = math.sqrt((menyique.x - munyeca.x)**2 + (menyique.y - munyeca.y)**2)

        # Verificar si solo los dedos índice y corazón están abiertos
        if (index_distance > apertura and ring_distance < cerrado and
            middle_distance > apertura and pinky_distance < cerrado and thumb_distance < 0.2):
            return True
        else:
            return False
        
    def is_open_hand():
        # Calcular las distancias entre los dedos
        thumb_distance = math.sqrt((pulgar.x - anular.x)**2 + (pulgar.y - anular.y)**2)
        index_distance = math.sqrt((indice.x - munyeca.x)**2 + (indice.y - munyeca.y)**2)
        middle_distance = math.sqrt((corazon.x - munyeca.x)**2 + (corazon.y - munyeca.y)**2)
        ring_distance = math.sqrt((anular.x - munyeca.x)**2 + (anular.y - munyeca.y)**2)
        pinky_distance = math.sqrt((menyique.x - munyeca.x)**2 + (menyique.y - munyeca.y)**2)

        # Verificar si todos los dedos están cerrados y cerca de la muñeca
        if (thumb_distance > 0.4 and
            index_distance > 0.5 and
            middle_distance > 0.5 and
            ring_distance > 0.5 and
            pinky_distance > 0.4):
            return True
        else:
            return False



    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Convertir el cuadro de BGR a RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detección de manos en el cuadro
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Nodos de la mano importantes
                    pulgar = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP] #dedo pulgar
                    indice = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP] #dedo indice
                    corazon = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP] #dedo corazon
                    anular = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP] #dedo anular
                    menyique = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP] #dedo meñique
                    munyeca = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST] #muñeca

                    # Dibujar los landmarks de la mano en el cuadro de video
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                            mp_drawing_styles.get_default_hand_landmarks_style(),
                                            mp_drawing_styles.get_default_hand_connections_style())
                    
                    if is_closed_fist():
                        cv2.putText(frame, gestures_labels[1], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        return gestures_labels[1]
                    if is_open_hand():
                        cv2.putText(frame, gestures_labels[2], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        return gestures_labels[2]
                    if is_victory():
                        cv2.putText(frame, gestures_labels[3], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        return gestures_labels[3]



            # Mostrar el cuadro de video con los landmarks de la mano y los gestos reconocidos
            cv2.imshow('Hand Gestures Detection', frame)
            # Salir del bucle si se presiona la tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Liberar la cámara
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()    





