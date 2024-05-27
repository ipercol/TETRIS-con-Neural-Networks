# TETRIS con Neural Networks

Instalación de librerias

#1.pip install pygame
#2.pip install opencv-python
#3.pip install mediapipe

# 
Este modelo del juego de Tetris utiliza las librerias de mediapipe studio y una red entrenada para detectar los 21 puntos de una mano
y calcular la distancia de una mano para así determinar los gestos de la mano. Estos son:

1. Puño cerrado = Rotar figura
2. Simbolo de la paz = Desplazar rapidamente hacia abajo
3. Mano abierta = Desplazamiento lateral, en funcion de si la mueves a la izquierda o derecha, se desplazara la pieza en esa dirección.

