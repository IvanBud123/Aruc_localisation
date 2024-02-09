import pygame as pg
from param import WIDTH, HEIGHT, WHITE, BLACK, BLUE, FPS
global screen, clock
pg.init()
pg.mixer.init()
screen = pg.display.set_mode((WIDTH, HEIGHT))

screen.fill(WHITE)
pg.display.set_caption("My Game")
clock = pg.time.Clock()
pg.draw.rect(screen, (252,122,0), (0,200,10,100))
pg.draw.rect(screen, (252,122,0), (200,0,100,10))
pg.display.update()