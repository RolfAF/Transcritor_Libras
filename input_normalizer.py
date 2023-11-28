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
                px, py = int(lm.x*1000), int(lm.y*1000)
                if id == 0:
                    lm.z = 0
                    ref_cx = cx
                    ref_cy = cy
                    ref_px = px
                    ref_py = py
                cz = lm.z*1000
                rel_cx = cx-ref_cx
                rel_cy = cy-ref_cy
                rel_px = px-ref_px
                rel_py = py-ref_py
                array.append([rel_px,rel_py,cz])
                cv2.circle(img, (int(425-cz), cy), 5, (139, 0, 0), cv2.FILLED)
                cv2.circle(img, (int(215+rel_cx), cy), 5, (139, 0, 0), cv2.FILLED)
            mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)
        #print(results.multi_hand_landmarks[0].landmark[0].y)
        for i in range(1,21):
            #print("i:",i)
            a = array[i][0]
            b = array[i][1]
            #if(a > 0):
            try:
                c2 = pow(a,2)+pow(b,2)
                c = math.sqrt(c2)
                array[i].append(c)
                #alpha = math.degrees(math.atan(a/b)) # 0 -> -90/90 -> 0 -> -90/90 -> 0
                alpha = math.degrees(math.acos(b/c)) # 0 -> 90 -> 180/180 -> 90 -> 0
                #print("c: ",c)
                #print("alpha: ",alpha)
                if(a > 0):
                    angle = 360 - alpha
                else:
                    angle = alpha
                #print("angle: ",angle)
                array[i].append(angle)
                #print(array[i])
            except Exception as e:
                print("Erro: ",e)
        ref_angle = 180 - array[9][4]
        print(ref_angle)
        for i in range(1,10): # <- mudar pra 21!
            print(ref_angle)
            target_angle = array[i][4] + ref_angle #correto!
            array[i].append(target_angle)
            new_a = array[i][3] * math.sin(math.radians(target_angle)) #correto?!!!! (transformar em zero para array[9])
            array[i].append(new_a)
            print(array[i])

                
        # calcular c e alpha para todos os pontos, subtrair alpha[9] dos outros pontos, c se torna b
        # para calcular novo x: a = c * sin(alpha)
        #https://www.omnicalculator.com/math/angle-of-right-triangle
        #https://www.omnicalculator.com/math/right-triangle-side-angle
        
        array = []
    # Time and FPS Calculation
    #print(results.multi_hand_landmarks.landmark[0][0])
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 3, (139,0,0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
