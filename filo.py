import pygame
import math
import random

n = 255
c = 8.5
concentracion = random.randint(60, 256)
sep = 0.63
stateEm = random.randint(1, 6)
circle = 1  # Cambiamos el valor inicial de circle a 1
red = 0
green = 0
blue = 0
radius = 0

pygame.init()

width, height = 1000, 1000
screen = pygame.display.set_mode((width, height))


def setup():
    global concentracion, stateEm, circle, red, green, blue
    pygame.display.set_caption("Python Processing")
    background_color = (concentracion / 255, concentracion / 255, concentracion / 255)
    screen.fill(background_color)
    pygame.display.update()
    print(stateEm)
    print(concentracion)

    red = 0.0
    green = 0.0
    blue = 0.0

    if stateEm == 1:
        red = 200.0
        green = 0.0
    elif stateEm == 2:
        red = 200.0
        blue = 0.0
    elif stateEm == 3:
        blue = 180.0
        green = 0.0
    elif stateEm == 4:
        blue = 180.0
        red = 0.0
    elif stateEm == 5:
        green = 180.0
        red = 0.0
    elif stateEm == 6:
        green = 180.0
        blue = 0.0

def dot(i, concentracion, stateEm):
    global red, green, blue, radius
    a = i * 2 * math.radians(concentracion)
    norm = concentracion * sep
    r = concentracion / norm * c * math.sqrt(i)
    x = concentracion / norm * r * math.cos(a)
    y = concentracion / norm * r * math.sin(a)

    if stateEm == 1:
        red += 1
        if red >= 255:
            red = 255
            green += 1
            if green >= 255:
                green = 255
    elif stateEm == 2:
        red += 1
        if red >= 255:
            red = 255
            blue += 1
            if blue >= 255:
                blue = 255
    elif stateEm == 3:
        green += 1
        if green >= 255:
            green = 255
            blue += 1.5
            if blue >= 255:
                blue = 255
    elif stateEm == 4:
        blue += 1
        if blue >= 255:
            blue = 255
            red += 2
            if red >= 255:
                red = 255
    elif stateEm == 5:
        green += 1
        if green >= 255:
            green = 255
            red += 1.5
            if red >= 255:
                red = 255
    elif stateEm == 6:
        green += 1
        if green >= 255:
            green = 255
            blue += 1.5
            if blue >= 255:
                blue = 255

    color = (int(red), int(green), int(blue))
    tam = 15
    tamy = 25
    radius = 11 + math.log(i * 4) * 3
    pygame.draw.ellipse(screen, color, (x - radius/2 + width/2, y - radius/2 + height/2, radius, radius))  # Ajustamos las coordenadas para centrar el círculo
    pygame.display.update()

def draw():
    global circle, concentracion, stateEm
    pygame.time.delay(30)  # Agrega un retraso para ver el dibujo gradualmente
    if circle <= n:  # Cambiamos el condicional para que incluya el primer círculo
        dot(circle, concentracion, stateEm)
        circle += 1
    else:
        setup()
        circle = 1  # Reiniciamos en 1 en lugar de 0 para que el primer círculo sea centrado
        concentracion = random.randint(120, 250)
        stateEm = random.randint(1, 6)
        draw()

if __name__ == "__main__":
    setup()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        draw()
