import pygame
import random
import sys
import cv2
import mediapipe as mp
import math
import numpy as np


# Inicializar Pygame
pygame.init()

# Configuración de MediaPipe
cap = cv2.VideoCapture(0)
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

# Definir constantes
SCREEN_WIDTH = 900  # Aumentamos el ancho de la pantalla
SCREEN_HEIGHT = 700
BLOCK_SIZE = 50
SCORE_MARGIN = 0  # Margen para el score
NEXT_PIECE_MARGIN = 5*50  # Margen para la próxima pieza
NEXT_PIECE_SIZE = BLOCK_SIZE  # Tamaño del área para mostrar la próxima pieza

# Calcular el número de cuadrículas en cada dirección
NUM_GRID_X = (SCREEN_WIDTH - NEXT_PIECE_MARGIN - NEXT_PIECE_SIZE) // BLOCK_SIZE  # Actualizamos el cálculo
NUM_GRID_Y = (SCREEN_HEIGHT - SCORE_MARGIN) // BLOCK_SIZE

# Definir piezas del Tetris
tetris_shapes = [
    [[1, 1, 1],
     [0, 1, 0]],

    [[0, 2, 2],
     [2, 2, 0]],

    [[3, 3, 0],
     [0, 3, 3]],

    [[4, 0, 0],
     [4, 4, 4]],

    [[0, 0, 5],
     [5, 5, 5]],

    [[6, 6, 6, 6]],

    [[7, 7],
     [7, 7]]
]

# Clase de la pieza del Tetris
class Piece:
    def __init__(self, x, y, shape):
        self.x = x
        self.y = y
        self.shape = shape
        self.color = random.choice([(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0), (255, 165, 0)])

# Inicializar ventana del juego
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Tetris")

#####################
####  INTERFACE  ####
#####################

# Función para mostrar la pantalla de inicio
def start_screen():
    background_image = pygame.image.load("images/wallPStart.jpg")
    background_image = pygame.transform.scale(background_image, (SCREEN_WIDTH, SCREEN_HEIGHT))

    tetris_logo = pygame.image.load("images/tetris_logo2.png")
    tetris_logo = pygame.transform.scale(tetris_logo, (int(tetris_logo.get_width() * 0.8), int(tetris_logo.get_height() * 0.8)))
    tetris_rect = tetris_logo.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 100))


    button_start = pygame.image.load("images/start3.png")
    button_start = pygame.transform.scale(button_start, (int(button_start.get_width() * 0.4), int(button_start.get_height() * 0.4)))
    button_rect = button_start.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 80))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                return

        screen.fill((255, 255, 255))
        screen.blit(background_image, (0, 0))
        screen.blit(tetris_logo, tetris_rect)
        screen.blit(button_start, button_rect)
        pygame.display.update()

# Función para mostrar la pantalla de Game Over
def game_over_screen(score):
    score_font = pygame.font.Font(None, 36)
    
    gameover_img = pygame.image.load("images/Game_Over2.png")
    gameover_rect = gameover_img.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))


    score_text = score_font.render(f"Score: {score}", True, (255, 255, 255))
    score_rect = score_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill((0, 0, 0))
        screen.blit(gameover_img, gameover_rect)
        screen.blit(score_text, score_rect)
        pygame.display.update()

# Contador antes de comenzar el juego
def countdown():
    background_image = pygame.image.load("images/wallPStart.jpg")
    background_image = pygame.transform.scale(background_image, (SCREEN_WIDTH, SCREEN_HEIGHT))

    imagen_cuenta = ["images/one.png", "images/two.png", "images/three.png"]

    for i in range(3, 0, -1):
        screen.fill((255, 255, 255))
        screen.blit(background_image, (0, 0))
        Num_image = pygame.image.load(imagen_cuenta[i-1])
        Num_rect = Num_image.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        screen.blit(Num_image, Num_rect)
        pygame.display.update()
        pygame.time.wait(800)

    # Mensaje "GO!" antes de comenzar el juego
    screen.fill((255, 255, 255))
    go_image = pygame.image.load("images/Go_text.png")
    go_image = pygame.transform.scale(go_image, (int(go_image.get_width() * 1.5), int(go_image.get_height() * 1.5)))
    go_rect = go_image.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
    screen.blit(background_image, (0, 0))
    screen.blit(go_image, go_rect)
    pygame.display.update()
    pygame.time.wait(800)

# Función para crear un tablero vacío
def create_board():
    return [[0 for _ in range(NUM_GRID_X)] for _ in range(NUM_GRID_Y)]

# Función para dibujar el tablero
def draw_board(surface, board):
    for y, row in enumerate(board):
        for x, color in enumerate(row):
            if color:
                pygame.draw.rect(surface, color, (x * BLOCK_SIZE, y * BLOCK_SIZE + SCORE_MARGIN, BLOCK_SIZE, BLOCK_SIZE), 0)
            pygame.draw.rect(surface, (128, 128, 128), (x * BLOCK_SIZE, y * BLOCK_SIZE + SCORE_MARGIN, BLOCK_SIZE, BLOCK_SIZE), 1)

# Función para crear una nueva pieza
def new_piece():
    shape = random.choice(tetris_shapes)
    return Piece(NUM_GRID_X // 2 - len(shape[0]) // 2, 0, shape)

# Función para comprobar si una pieza puede moverse
def valid_space(board, piece):
    for y, row in enumerate(piece.shape):
        for x, val in enumerate(row):
            if val and (piece.y + y >= len(board) or piece.x + x < 0 or piece.x + x >= len(board[0]) or board[piece.y + y][piece.x + x]):
                return False
    return True

# Función para fijar una pieza en el tablero
def fix_piece(board, piece):
    for y, row in enumerate(piece.shape):
        for x, val in enumerate(row):
            if val:
                board[piece.y + y][piece.x + x] = piece.color

# Función para borrar filas completas y calcular la puntuación
def clear_rows(board):
    rows_to_clear = [i for i, row in enumerate(board) if all(row)]
    num_rows_cleared = len(rows_to_clear)
    for row in rows_to_clear:
        del board[row]
        board.insert(0, [0 for _ in range(NUM_GRID_X)])
    return num_rows_cleared


#####################
# GESTOS DE LA MANO #
#####################

def is_closed_fist(pulgar, indice, corazon, anular, menyique, munyeca):
    threshold=0.25
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

def is_victory(pulgar, indice, corazon, anular, menyique, munyeca):
    apertura=0.4
    cerrado = 0.2
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
    
def is_open_hand(pulgar, indice, corazon, anular, menyique, munyeca):
    large =0.4
    small = 0.3
    # Calcular las distancias entre los dedos
    thumb_distance = math.sqrt((pulgar.x - anular.x)**2 + (pulgar.y - anular.y)**2)
    index_distance = math.sqrt((indice.x - munyeca.x)**2 + (indice.y - munyeca.y)**2)
    middle_distance = math.sqrt((corazon.x - munyeca.x)**2 + (corazon.y - munyeca.y)**2)
    ring_distance = math.sqrt((anular.x - munyeca.x)**2 + (anular.y - munyeca.y)**2)
    pinky_distance = math.sqrt((menyique.x - munyeca.x)**2 + (menyique.y - munyeca.y)**2)

    # Verificar si todos los dedos están cerrados y cerca de la muñeca
    if (thumb_distance > small and
        index_distance > large and
        middle_distance > large and
        ring_distance > large and
        pinky_distance > small):
        return True
    else:
        return False
    

#####################
###  MOVIMIENTOS  ###
#####################

def acelerate_piece(piece, board):
    piece.y += 1
    if not valid_space(board, piece):
        piece.y -= 1

def horizontal_move(piece, board, left_region, right_region, hand_position_x):
    # Verificar si la muñeca está en la región izquierda y la pieza no está en el límite izquierdo
    if left_region[0] <= hand_position_x < left_region[1] and piece.x > 0:
        # Mover la pieza hacia la izquierda
        piece.x -= 1
        if not valid_space(board, piece):
                piece.x += 1  # Deshacer el movimiento si hay colisión

    # Verificar si la muñeca está en la región derecha y la pieza no está en el límite derecho
    elif right_region[0] <= hand_position_x < right_region[1] and piece.x + len(piece.shape[0]) < NUM_GRID_X:
        # Mover la pieza hacia la derecha
        piece.x += 1
        if not valid_space(board, piece):
                piece.x -=1


# Función principal del juego
def main():
    start_screen()
    pygame.time.wait(100)
    countdown()

    board = create_board()
    piece = new_piece()
    next_piece = new_piece()  # Agregar la próxima pieza
    clock = pygame.time.Clock()
    fall_time = 0
    fall_speed = 0.8
    rotate_speed = 0.8
    horizontal_speed = 0.5
    last_rotate_time = pygame.time.get_ticks()  # Tiempo del último intento de rotación
    last_horizontal_time = pygame.time.get_ticks()  # Tiempo del último intento de rotación
    game_over = False
    score = 0

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:    

        while not game_over:

            screen.fill((255, 255, 255))
            fall_time += clock.get_rawtime()
            clock.tick()

            # Capturar frame de la cámara y procesar gestos
            success, frame = cap.read()
            frame = cv2.flip(frame, 1)
            if success:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                                mp_drawing_styles.get_default_hand_connections_style())
                        
                        if is_closed_fist(pulgar, indice, corazon, anular, menyique, munyeca):
                            cv2.putText(frame, gestures_labels[1], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                            # Verificar si ha pasado suficiente tiempo desde la última rotación
                            current_time = pygame.time.get_ticks()
                            if current_time - last_rotate_time > rotate_speed*1000:
                                # Rotar la pieza
                                piece_shape = piece.shape[:]
                                piece.shape = [list(row) for row in zip(*piece_shape[::-1])]
                                if not valid_space(board, piece):
                                    piece.shape = piece_shape
                                # Actualizar el tiempo del último intento de rotación
                                last_rotate_time = current_time
                            
                        if is_open_hand(pulgar, indice, corazon, anular, menyique, munyeca):
                            cv2.putText(frame, gestures_labels[2], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

                            # Calcular el centro de la cámara como punto de referencia
                            camera_center = SCREEN_WIDTH // 2
                            # Verificar si ha pasado suficiente tiempo desde el último movimiento horizontal
                            current_time = pygame.time.get_ticks()

                            # Calcular la posición horizontal de la muñeca
                            hand_position_x = int(munyeca.x * SCREEN_WIDTH)
                            # Definir las regiones izquierda y derecha
                            left_region = (0, camera_center - 60)
                            right_region = (camera_center + 60, SCREEN_WIDTH)

                            if current_time - last_horizontal_time > horizontal_speed*1000:  
                                horizontal_move(piece, board, left_region, right_region, hand_position_x)
                                last_horizontal_time = current_time  # Actualizar el tiempo del último movimiento lateral


                        if is_victory(pulgar, indice, corazon, anular, menyique, munyeca):
                            cv2.putText(frame, gestures_labels[3], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                            acelerate_piece(piece, board)
                            
                    
                cv2.imshow('Hand Gestures Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


            #Control de Eventos. IMPORTANTE
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_over = True
                
            # Mover la pieza hacia abajo
            if fall_time / 1000 >= fall_speed:
                fall_time = 0
                piece.y += 1
                if not valid_space(board, piece):
                    piece.y -= 1
                    fix_piece(board, piece)
                    num_rows_cleared = clear_rows(board)
                    score += num_rows_cleared * 100  # Incrementar la puntuación
                    piece = next_piece  # Actualizar la pieza actual
                    next_piece = new_piece()  # Generar una nueva próxima pieza
                    if not valid_space(board, piece):
                        game_over = True

            # Dibujar la pieza y el tablero
            for y, row in enumerate(piece.shape):
                for x, val in enumerate(row):
                    if val:
                        pygame.draw.rect(screen, piece.color, ((piece.x + x) * BLOCK_SIZE, (piece.y + y) * BLOCK_SIZE + SCORE_MARGIN, BLOCK_SIZE, BLOCK_SIZE), 0)
                        pygame.draw.rect(screen, (128, 128, 128), ((piece.x + x) * BLOCK_SIZE, (piece.y + y) * BLOCK_SIZE + SCORE_MARGIN, BLOCK_SIZE, BLOCK_SIZE), 1)

            # Dibujar la próxima pieza en el margen derecho
            for y, row in enumerate(next_piece.shape):
                for x, val in enumerate(row):
                    if val:
                        pygame.draw.rect(screen, next_piece.color, ((NUM_GRID_X + 1 + x) * BLOCK_SIZE, (y + 1) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 0)
                        pygame.draw.rect(screen, (128, 128, 128), ((NUM_GRID_X + 1 + x) * BLOCK_SIZE, (y + 1) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 1)

            draw_board(screen, board)

            # Mostrar la puntuación
            font = pygame.font.Font(None, 36)
            score_text = font.render(f"Score: {score}", True, (0, 0, 0))
            score_rect = score_text.get_rect(midright=(SCREEN_WIDTH - (NEXT_PIECE_MARGIN//2), SCREEN_HEIGHT // 2))
            screen.blit(score_text, score_rect)

            pygame.display.update()

    # Llamar a la función game_over_screen con la puntuación final
    game_over_screen(score)


if __name__ == "__main__":
    main()
    
