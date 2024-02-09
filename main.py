from cv2 import aruco
import numpy as np
import time
import cv2 as cv2
import cv2.aruco as aruco
import os
import openpyxl
import pandas as pd
import math
import pygame as pg
import sys
from statistics import median
from scipy.optimize import root_scalar


exec(open('param.py').read())
exec(open('arcopy.py').read())
exec(open('pgame.py').read())
import arcopy
import pgame
import tabl

global id21, id22, x1 ,y1,x2,y2,Mx,My
x1 = int(125)
y1 = int(1)
x2 = int(1)
y2 = int(125)
Mx = x1/2
My = y2/2
##id21 = int (input("введите маркер 1: "))
##id22 = int (input("введите маркер 2: "))
id21 = 13
id22 = 5
coef1 = False
coef2 = False
start1 = time.time()
def read_vid():
    global ret, frame
    ret, frame = cap.read()  # ЧИТАЕМ ВИДЕОПОТОК

    global start2
    start2 = time.time()
    #                                                    Convert to grayscale
    global gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #                                                Detect ArUco markers

    global corners, ids, rejected
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
def teme():
    global time1, toc1, tic2
    time1 = (start2 - start1) // 1
    if time1 != toc1:
        tic2 = True
        print(time1)
    if time1 == toc1:
        tic2 = False
    toc1 = time1
def id1():
    global coord1, rvecs1, tvecs1, distance1, rmtrix1, theta_x1, theta_y1, theta_y_arr_1, distance_arr1
    coord1 = corners[0][0][0]  # ОПРЕДЕЛЯЕМ ПЕРЕМЕННУЮ CORD1 КАК ПЕРВЫЙ МАРКЕР
    rvecs1, tvecs1, _ = aruco.estimatePoseSingleMarkers(corners[0], marker_size, camera_matrix,dist_coeffs)  # РАСЧИТЫВАЕМ RVEC И TVEC ПО RVEC МЫ БУДЕМ СЧИТАТЬ УГОЛ, A ПО TCEC РАССТОЯНИЕ
    distance1 = np.linalg.norm(tvecs1)  # РАСИТЫВАЕМ РАССТОЯНИЕ ДО МАРКЕРА
    distance1 = distance1 * 2  # кАЛИБРУЕМ РАСТОЯНИЕ
    rmtrix1, _ = cv2.Rodrigues(rvecs1)  # СОЗДАЁМ МАТРИЦУ ТРОПЕЦЫЙ ПО КОТОРОЙ БУДЕМ ОПРЕДЕЛЯТЬ УГЛЫ
    # ОПРЕДЕЛЯЕМ УГЛЫ X, Y В РАДИАНАХ
    theta_x1 = np.arctan2(rmtrix1[2, 1], rmtrix1[2, 2])
    theta_y1 = np.arctan2(-rmtrix1[2, 0], np.sqrt(rmtrix1[2, 1] ** 2 + rmtrix1[2, 2] ** 2))
    # ПЕРЕВОДИМ РАДИАНЫ В БОЛЕЕ ПОНЯТНЫЕ ДЛЯ НАС СМЕРТНЫХ ГРАДУСЫ
    theta_x1 = np.degrees(theta_x1)
    theta_y1 = np.degrees(theta_y1)
    # ДОБАВЛЯЕМ УГОЛ И РАССТОЯНИЕ В МАССИВ ДЛЯ МЕДИАННОГО УСРЕДНЕНИЯ
    theta_y_arr1.append(theta_y1)
    distance_arr1.append(distance1)
    # ПРИСВАИВАЕМ НОМЕР ПЕРВОМУ МАРКЕРУ
    coord1 = tuple(map(int, coord1))
    for i in range(rvecs1.shape[0]):
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs1[i, :, :], tvecs1[i, :, :], 0.015)
def median1():
    if len(theta_y_arr1) >= 200:
        global theta_x1, coef1
        theta_x1= np.median(theta_x_arr1[-100:])
        global theta_y1
        theta_y1 = np.median(theta_y_arr1[-100:])
        global distance1
        distance1 = abs(np.median(distance_arr1[-200:]))
        coef1 = True
def outp1():
    cv2.putText(frame, f"dist1: {distance1 * 100:.2f} cm", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f"y1: {int(theta_y1):.2f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
global coord2 , rvecs2, tvecs2, distance2, distance12, theta_x2, theta_y2, theta_y_arr2, distance_arr2, distance_arr12, tech1, tech2, coord1, rvecs1, tvecs1, distance1, rmtrix1, theta_x1, theta_y1, theta_y_arr_1, distance_arr1
def id2():
    tech1 = False
    tech2 = False
    for i in range(0, len(ids)):
        if ids[i] == id21:
            # ДЕЛАЕМ АНАЛАГИЧНЫЕ ПРОЦЕДУРЫ ЧТО И В СТРОЧКАХ 194-224
            coord1 = corners[0][0][0]
            rvecs1, tvecs1, _ = aruco.estimatePoseSingleMarkers(corners[0], marker_size, camera_matrix, dist_coeffs)
            rmtrix1, _ = cv2.Rodrigues(rvecs1)
            theta_x1 = np.arctan2(rmtrix1[2, 1], rmtrix1[2, 2])
            theta_y1 = np.arctan2(-rmtrix1[2, 0], np.sqrt(rmtrix1[2, 1] ** 2 + rmtrix1[2, 2] ** 2))
            theta_x1 = np.degrees(theta_x1)
            theta_y1 = np.degrees(theta_y1)
            theta_y1 = abs(theta_y1)
            theta_y_arr1.append(theta_y1)
            distance1 = np.linalg.norm(tvecs1)
            distance1 = distance1 * 2
            distance_arr1.append(distance1)
            tech2 = True

        if ids[i] == id22 and tech2 ==True:
            coord2 = corners[1][0][0]
            rvecs2, tvecs2, _ = aruco.estimatePoseSingleMarkers(corners[1], marker_size, camera_matrix, dist_coeffs)
            distance2 = np.linalg.norm(tvecs2)
            distance12 = np.linalg.norm(tvecs1 - tvecs2)
            distance12 = (distance12 + 0.007)
            distance2 = distance2 * 2

            rmtrix2, _ = cv2.Rodrigues(rvecs2)
            theta_x2 = np.arctan2(rmtrix2[2, 1], rmtrix2[2, 2])
            theta_y2 = np.arctan2(-rmtrix2[2, 0], np.sqrt(rmtrix2[2, 1] ** 2 + rmtrix2[2, 2] ** 2))
            theta_x2 = np.degrees(theta_x2)
            theta_y2 = np.degrees(theta_y2)
            theta_y_arr2.append(theta_y2)
            distance_arr2.append(distance2)

            distance_arr12.append(distance12)
            coord2 = tuple(map(int, coord2))
            theta_y2 = abs(theta_y2)
            tech1 = True

            for i in range(rvecs2.shape[0]):
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs2[i, :, :], tvecs2[i, :, :], 0.015)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs1[i, :, :], tvecs1[i, :, :], 0.015)
            if tech1 == True:
                cv2.putText(frame, f"dist1: {distance1 * 100:.2f} cm", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 1)
                cv2.putText(frame, f"y1: {int(theta_y1):.2f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),1)
                cv2.putText(frame, f"dist2: {distance2 * 100:.2f} cm", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 1)
                cv2.putText(frame, f"y2: {abs(int(theta_y2)):.2f} deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, f"dist 1&2: {distance12*100:.2f} cm", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 0, 0), 1)
def median2():
    global theta_x2, theta_y2, distance2, distance12, coef2
    if len(theta_y_arr1) >= 200 and len(theta_y_arr2) >= 200 and len(distance_arr12) >= 200 and len(distance_arr2) >= 200:
        theta_x2 = np.median(theta_x_arr2[-200:])
        theta_y2 = np.median(theta_y_arr2[-200:])
        distance2 = abs(np.median(distance_arr2[-200:]))
        distance12 = abs(np.median(distance_arr12[-200:]))
        coef2 = True


def geo1():
    global X1, Y1, thi1, angle, X2,Y2
    x1, y1 = 15,0
    r1 = distance1*100
    # Параметры второй окружности
    x2, y2 = 0, 15
    r2 = distance2*100

    # Вычисление расстояния между центрами окружностей
    distance12 = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    distance12 = 23
    # Проверка на пересечение окружностей
    if r1*100 + r2*100 >= distance12:
        # Находим be
        be = ((r1 ** 2 - r2 ** 2 + distance12 ** 2) / (2 * distance12))

        # Находим координаты точки пересечения на отрезке между центрами окружностей
        px1 = x1 + (be * (x2 - x1) / distance12)
        py1 = y1 + (be * (y2 - y1) / distance12)

        # Находим высоту от точки пересечения до точки пересечения
        he = ((r1 ** 2 - be ** 2) ** 0.5)

        # Находим две точки пересечения окружностей
        X1 = px1 + (he * (y2 - y1) / distance12)
        Y1 = py1 - (he * (x2 - x1) / distance12)
        X2 = px1 - (he * (y2 - y1) / distance12)
        Y2 = py1 + (he * (x2 - x1) / distance12)

        print(f"Первая точка пересечения: X1 = {X1}, Y1 = {Y2}")
#        print(f"Вторая точка пересечения: X2 = {X2}, Y2 = {Y2}")
    else:
        print("Окружности не пересекаются.")

    #print("angle = ", angle)
    cv2.putText(frame, f"X: {X1:.2f} cm", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f"Y: {Y2:.2f} deg", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
def drow1():
    screen.fill(WHITE)
    pg.display.flip()
    pg.draw.circle(screen,(255,0,0), (200,0), distance1*1000, 2)
    pg.draw.circle(screen,(255, 0, 0), (0, 200), distance2*1000, 2)
    pg.draw.circle(screen, (25,122,0), (abs(Y2*11+11),abs(X1*11+11)),30,10)
    pg.draw.rect(screen, (252,122,0), (0,200,10,100))
    pg.draw.rect(screen, (252,122,0), (200,0,100,10))
    pg.display.update()
while True:
    clock.tick(FPS)  # УСИАНАВЛИВКАЕМ ЛОК НА FPS
    start2 = time.time()
    teme()  
    read_vid()
    pg.draw.circle(screen,(1,0,0),(10,10),1,2)
    pg.display.update()
    if coef1 ==True:
        pg.draw.rect(screen, (255,255,255),(0,0,20,20))
        pg.display.update()
    if ids is not None and len(ids) ==1:
        id1()
        median1()
        outp1()
    if ids is not None and len(ids) == 2:
        id2()
        median1()
        median2()
        #outp1()

        if coef1 == True and coef2 ==True:

            #if tic2 == True:
            geo1()
            drow1()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('e'):
        marker_size = int(input("Marker_size_in_mm:  ")) / (10 ** 3)
        FPS = int(input("FPS:  "))
cap.release()
cv2.destroyAllWindows()
