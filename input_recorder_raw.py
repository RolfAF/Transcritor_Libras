import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
line = ''
cont = 0
cTime = 0
pTime = 0# Function Start
time.sleep(1)
inp = input("Iniciar leitura\n")
while(cont <= 100):
    line = str(cont)+":"
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
                print(id, lm.x, lm.y, lm.z)
                line += " "+str(lm.x)+" "+str(lm.y)+" "+str(lm.z)
                #if id == 5:
                cv2.circle(img, (int(425-cz), cy), 5, (139, 0, 0), cv2.FILLED)
                cv2.circle(img, (int(215+rel_x), cy), 5, (139, 0, 0), cv2.FILLED)
            mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)
        file_a.write(line+"\n")
        file_a.close()
    # Time and FPS Calculation
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 3, (139,0,0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
    cont += 1
