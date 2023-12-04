import cv2
import mediapipe as mp
import time
import numpy as np
import math

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1,model_complexity=1,min_detection_confidence=0.9,min_tracking_confidence=0.9)
mpDraw = mp.solutions.drawing_utils
cTime = 0
pTime = 0# Function Start
array = []
gesture = 0
line = ''
cont = 0
time.sleep(1)
#inp = input("Iniciar leitura\n")
while True:
    line = ''
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        file_a = open("a.txt","a")
        for handlms in results.multi_hand_landmarks:
            for id, lm in enumerate(handlms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                #px, py = int(lm.x*1000), int(lm.y*1000)
                if id == 0:
                    lm.z = 0
                    ref_cx = cx
                    ref_cy = cy
                    #ref_px = px
                    #ref_py = py
                cz = lm.z*1500
                rel_cx = cx-ref_cx
                rel_cy = cy-ref_cy
                #rel_px = px-ref_px
                #rel_py = py-ref_py
                array.append([rel_cx,rel_cy,cz])

                #cv2.circle(img, (int(215+rel_cx), cy), 5, (139, 0, 0), cv2.FILLED)
            mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)
        #print(results.multi_hand_landmarks[0].landmark[0].y)
        a = array[9][0]
        b = array[9][1]
        c2 = math.pow(a,2) + math.pow(b,2)
        c = math.sqrt(c2)
        #array[i].append(c)
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
        for i in range(1,21):
            x = array[i][0] / ref_scale
            y = array[i][1] / ref_scale
            x2 = aux_x + (x*math.cos(ref_angle)) - (y*math.sin(ref_angle))
            y2 = aux_y + (x*math.sin(ref_angle)) + (y*math.cos(ref_angle))
            array[i].append(x2)
            array[i].append(y2)
            print(array[i])
            cv2.circle(img, (int(array[i][3]), int(array[i][4])), 5, (139, 0, 0), cv2.FILLED)
            cv2.circle(img, (int(350-array[i][2]), int(aux_y + y)), 5, (139, 0, 0), cv2.FILLED)
        array = []
    # Time and FPS Calculation
    #print(results.multi_hand_landmarks.landmark[0][0])
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 3, (139,0,0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

#https://en.wikipedia.org/wiki/Rotation_matrix
#https://www.geeksforgeeks.org/2d-transformation-in-computer-graphics-set-1-scaling-of-objects/
