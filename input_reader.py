import cv2
import mediapipe as mp
import time
import numpy as np
import math

#cap = cv2.VideoCapture(0)
#mpHands = mp.solutions.hands
#hands = mpHands.Hands(max_num_hands=1,model_complexity=1,min_detection_confidence=0.9,min_tracking_confidence=0.9)
#mpDraw = mp.solutions.drawing_utils
cTime = 0
pTime = 0# Function Start
array = []
gesture = 0
line = ''
cont = 0
time.sleep(1)
aux_lines = []
point = []
points = []
h = 480
w = 640
#inp = input("Iniciar leitura\n")
f = open("raw/raw_a.txt", "r")
lines = f.readlines()
#print(lines)
#print(lines[0])
for i in range(0,len(lines)): #separa todos os elementos de cada linha
    img = np.zeros((h,w,3), np.uint8)
    lines[i] = lines[i].split(",")
    for j in range(0,60,3): #agrupa as coordenadas x, y e z de cada ponto, e agrupa os 20 pontos
        point.append(float(lines[i][j]))
        point.append(float(lines[i][j+1]))
        point.append(float(lines[i][j+2]))
        points.append(point)
        point = []
    lines[i] = points #guarda os pontos na linha
    points = []

    a = lines[i][10][0]
    b = lines[i][10][1]
    c2 = math.pow(a,2) + math.pow(b,2)
    c = math.sqrt(c2)
    print("c:",c)
    try:
        #alpha = math.degrees(math.atan(a/b)) # 0 -> -90/90 -> 0 -> -90/90 -> 0
        alpha = math.degrees(math.acos(b/c)) # 0 -> 90 -> 180/180 -> 90 -> 0
        if(a > 0):
            angle = 360 - alpha
        else:
            angle = alpha
        ref_angle = 180 - angle
    except Exception as e:
        print("Erro: ",e)
    print("angle:",angle)
    aux_x = 200
    aux_y = h - 10
    ref_angle = math.radians(ref_angle)
    ref_scale = c / 200
    cv2.circle(img, (200, aux_y), 5, (139, 0, 0), cv2.FILLED)
    cv2.circle(img, (350, aux_y), 5, (139, 0, 0), cv2.FILLED)
    for j in range(0,20):
        x = lines[i][j][0] / ref_scale
        y = lines[i][j][1] / ref_scale
        x2 = aux_x + (x*math.cos(ref_angle)) - (y*math.sin(ref_angle))
        y2 = aux_y + (x*math.sin(ref_angle)) + (y*math.cos(ref_angle))
        lines[i][j][0] = int(x2)
        lines[i][j][1] = int(y2)
        #array[j].append(x2)
        #array[j].append(y2)
        print(lines[i][j])
        cv2.circle(img, (int(lines[i][j][0]), int(lines[i][j][1])), 5, (139, 0, 0), cv2.FILLED)
        cv2.circle(img, (int(350-float(lines[i][j][2])), int(aux_y + y)), 5, (139, 0, 0), cv2.FILLED)
    #array = []
    cv2.imshow("Image", img)
    cv2.waitKey(1)
#cv2.circle(img, (int(array[i][3]), int(array[i][4])), 5, (139, 0, 0), cv2.FILLED)
