import cv2
import mediapipe as mp
import time
import numpy as np

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
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        file_a = open("a.txt","a")
        for handlms in results.multi_hand_landmarks:
            for id, lm in enumerate(handlms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                if id == 0:
                    lm.z = 0
                    ref_x = cx
                    ref_y = cy
                #print(id, lm)
                #cx, cy = lm.x, lm.y
                cz = lm.z*1000
                rel_x = int(cx-ref_x)
                rel_y = int(cy-ref_y)
                #print(id, rel_x, rel_y, int((w/2)-cz))
                #if id == 5:
                cv2.circle(img, (int(425-cz), cy), 5, (139, 0, 0), cv2.FILLED)
                cv2.circle(img, (int(215+rel_x), cy), 5, (139, 0, 0), cv2.FILLED)
                array.append(int(cx))
                array.append(int(cy))
                array.append(int(cz))
                line += str(int(cx))+","+str(int(cz))+","+str(int(cy))
            mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)
        #print(results.multi_hand_landmarks[0].landmark[0].x)
        if(len(array) == 63):
            array.append(gesture)
            line += ","+str(gesture)
            hand_np_array = np.array(array)
            print(line)
            file_a.write(line+"\n")
            file_a.close()
        array = []
        #print(array)
    # Time and FPS Calculation
    #print(results.multi_hand_landmarks.landmark[0][0])
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 3, (139,0,0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
