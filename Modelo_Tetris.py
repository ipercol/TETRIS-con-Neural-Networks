import pygame
import random
import sys

# Inicializar Pygame
pygame.init()

# Definir constantes
SCREEN_WIDTH = 700  # Aumentamos el ancho de la pantalla
SCREEN_HEIGHT = 700
BLOCK_SIZE = 40
SCORE_MARGIN = 50  # Margen para el score
NEXT_PIECE_MARGIN = 50  # Margen para la próxima pieza
NEXT_PIECE_SIZE = 4 * BLOCK_SIZE  # Tamaño del área para mostrar la próxima pieza

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

# Función para mostrar la pantalla de inicio
def start_screen():
    start_font = pygame.font.Font(None, 64)
    start_text = start_font.render("TETRIS", True, (255, 0, 0))
    start_rect = start_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))

    button_font = pygame.font.Font(None, 36)
    button_text = button_font.render("Start", True, (255, 255, 255))
    button_rect = button_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                return

        screen.fill((0, 0, 0))
        screen.blit(start_text, start_rect)
        pygame.draw.rect(screen, (0, 0, 255), button_rect, 2)
        screen.blit(button_text, button_rect)
        pygame.display.update()

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

# Función para mostrar la pantalla de Game Over
def game_over_screen(score):
    game_over_font = pygame.font.Font(None, 64)
    score_font = pygame.font.Font(None, 36)
    
    game_over_text = game_over_font.render("Game Over", True, (255, 0, 0))
    game_over_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))

    score_text = score_font.render(f"Score: {score}", True, (255, 255, 255))
    score_rect = score_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill((0, 0, 0))
        screen.blit(game_over_text, game_over_rect)
        screen.blit(score_text, score_rect)
        pygame.display.update()

# Función principal del juego
def main():
    start_screen()
    pygame.time.wait(1000)

    # Contador antes de comenzar el juego
    for i in range(3, 0, -1):
        screen.fill((255, 255, 255))
        font = pygame.font.Font(None, 100)
        text = font.render(str(i), True, (255, 0, 0))
        text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        screen.blit(text, text_rect)
        pygame.display.update()
        pygame.time.wait(1000)

    # Mensaje "READY!" antes de comenzar el juego
    screen.fill((255, 255, 255))
    font = pygame.font.Font(None, 100)
    text = font.render("READY!", True, (255, 0, 0))
    text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
    screen.blit(text, text_rect)
    pygame.display.update()
    pygame.time.wait(1000)

    board = create_board()
    piece = new_piece()
    next_piece = new_piece()  # Agregar la próxima pieza
    clock = pygame.time.Clock()
    fall_time = 0
    fall_speed = 0.5
    game_over = False
    score = 0

    while not game_over:
        screen.fill((255, 255, 255))

        fall_time += clock.get_rawtime()
        clock.tick()

        # Manejar eventos de teclado
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    piece.x -= 1
                    if not valid_space(board, piece):
                        piece.x += 1
                elif event.key == pygame.K_RIGHT:
                    piece.x += 1
                    if not valid_space(board, piece):
                        piece.x -= 1
                elif event.key == pygame.K_DOWN:
                    piece.y += 1
                    if not valid_space(board, piece):
                        piece.y -= 1
                elif event.key == pygame.K_UP:
                    # Rotar la pieza
                    piece_shape = piece.shape[:]
                    piece.shape = [list(row) for row in zip(*piece_shape[::-1])]
                    if not valid_space(board, piece):
                        piece.shape = piece_shape

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
        score_rect = score_text.get_rect(midright=(SCREEN_WIDTH - 20, SCREEN_HEIGHT // 2))
        screen.blit(score_text, score_rect)

        pygame.display.update()

    # Llamar a la función game_over_screen con la puntuación final
    game_over_screen(score)

if __name__ == "__main__":
    main()
