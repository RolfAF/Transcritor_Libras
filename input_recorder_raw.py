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
array = []
gesture = 20
filename = "raw_y"
time.sleep(1)
inp = input("Iniciar leitura\n")
while(cont < 1100):
    #line = str(cont)+":"
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        file_ = open(filename,"a")
        for handlms in results.multi_hand_landmarks:
            for id, lm in enumerate(handlms.landmark):
                h, w, c = img.shape
                cx, cy = lm.x*w, lm.y*h
                px, py = lm.x*1000, lm.y*1000
                if id == 0:
                    lm.z = 0
                    ref_x = cx
                    ref_y = cy
                    ref_px = px
                    ref_py = py
                #print(id, lm)
                #cx, cy = lm.x, lm.y
                cz = lm.z*1000
                rel_x = int(cx-ref_x)
                rel_y = int(cy-ref_y)
                rel_px = px-ref_px
                rel_py = py-ref_py
                #print(id, lm.x, lm.y, lm.z)
                if id > 0:
                    line += str(rel_px)+","+str(rel_py)+","+str(cz)+","
                array.append(cx)
                array.append(cy)
                array.append(cz)
                cv2.circle(img, (int(425-cz), int(cy)), 5, (139, 0, 0), cv2.FILLED)
                cv2.circle(img, (int(215+rel_x), int(cy)), 5, (139, 0, 0), cv2.FILLED)
            mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)
        if(len(array) == 63):
            array.append(gesture)
            line += str(gesture)
            #hand_np_array = np.array(array)
            #print(line)
            if(cont > 99):
                file_.write(line+"\n")
                file_.close()
            cont += 1
            print(cont-99,array)
        array = []
        line = ''
    # Time and FPS Calculation
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 3, (139,0,0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
