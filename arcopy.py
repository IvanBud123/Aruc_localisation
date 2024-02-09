from cv2 import aruco
import cv2 as cv2
import cv2.aruco as aruco
def arc():
    global cap
    cap = cv2.VideoCapture(-1)
    #cap = cv2.VideoCapture("")
#                            Load ArUco dictionary

    global aruco_dict
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
#                          Create ArUco parameters

    global  aruco_params
    aruco_params = cv2.aruco.DetectorParameters()
arc()
