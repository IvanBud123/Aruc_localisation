import numpy as np
global tech1,tech2, camera_matrix, dist_coeffs,newcameramtx,theta_x_arr1,theta_y_arr1,theta_x_arr2,theta_y_arr2,distance_arr1, distance_arr2,theta_x_arr3,theta_y_arr3,distance_arr3

tech1 =False
tech2 =False
global WIDTH, HEIGHT, FPS, WHITE,BLACK,RED,GREEN, BLUE
# задаём размеры окна вывода
WIDTH = 100 *10
HEIGHT = 100 *10
FPS = 30000

# Задаем цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

#                                Set camera parameters
camera_matrix = np.array([[398.12724231  , 0.      ,   304.35638757],
 [  0.       ,  345.38259888, 282.49861858],
 [  0.,           0.,           1.        ]])

dist_coeffs =np.array(([[-0.58650416 , 0.59103816, -0.00443272 , 0.00357844 ,-0.27203275]]))

newcameramtx=np.array([[189.076828   ,  0.    ,     361.20126638]
 ,[  0 ,2.01627296e+04 ,4.52759577e+02]
 ,[0, 0, 1]])

#                           ARRAY OF FLOATS TO MEDIAN SORT

theta_x_arr1 = []
theta_y_arr1 = []
distance_arr1 = []

theta_x_arr2 = []
theta_y_arr2 = []
distance_arr2 = []

theta_x_arr3 = []
theta_y_arr3 = []
distance_arr3 = []

global distance_arr12, distance_arr13, distance_arr23, xA_arr, yA_arr, x11_arr,y11_arr,x12_arr,y12_arr,x13_arr,y13_arr

distance_arr12= []
distance_arr23= []
distance_arr13= []

yA_arr =[]
xA_arr=[]

x11_arr=[]
y11_arr=[]
x12_arr=[]
y12_arr=[]
x13_arr=[]
y13_arr=[]

# ЗАНУЛЕНИЕ НУЖНЫХ ПЕРЕМЕННЫХ(ОНИ В БУДУЮЩЕМ БУДУТ НУЖНЫ ДЛЯ ФИЛЬТРАЦИИ)

global AH1Cx1, AH1Cx2, AH2By1, AH2By2

AH1Cx1 = 0
AH1Cx2 = 0
AH2By1 = 0
AH2By2 = 0
global tic
tic =1
global toc1, AH1C, AH2B,inq, marker_size, xb,yb,x2,y2,chep1
toc1 =456
AH1C = 0
AH2B = 0
inq =2
#    ПАРАМЕТРЫ МАРКЕРОВ
marker_size = 0.060
#    ПОЛОЖЕНИЕ МАРКЕРОВ
xb= 2
yb = 3
x2 = 0
y2 = 0
chep1 =False