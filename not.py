import pygame
import random
import Modelo_Mano
import sys
import cv2

# Configuración de la captura de video desde la cámara web
cap = cv2.VideoCapture(0)

Modelo_Mano.main(cap)




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