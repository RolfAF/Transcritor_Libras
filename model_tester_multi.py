from multiprocessing import Process, Manager, Value
import cv2
import mediapipe as mp
import time
import numpy as np
import math
from tensorflow.keras.models import load_model
import pandas as pd

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1,model_complexity=1,min_detection_confidence=0.9,min_tracking_confidence=0.9)
mpDraw = mp.solutions.drawing_utils
#manager = multiprocessing.Manager()
#manager.start()
cTime = 0
pTime = 0# Function Start
array = []
sample = []
sample.append([])
gesture = 0
line = ''
cont = 0
last_p = 1
time.sleep(1)
prefix = ["a","b","c","d","e","f","g","i","l","m","n","o","p","q","r","s","t","u","v","w","x","y"]
model = load_model('model_3')
#sample_array = Manager().list([Manager().list()])
sample_array = Manager().list()
#model1 = load_model('model_3')
#model2 = load_model('model_3')
#model3 = load_model('model_3')
def prediction(sample_array):
    print("asd")
    #from tensorflow.keras.models import load_model
    model = load_model('model_3')
    sample = []
    while True:
        #print("process: ",len(sample_array))
        if len(sample_array) > 0:
            sample = sample_array[0]
            sample_array.pop(0)
            #print(sample)
            predictions = model.predict(pd.DataFrame(sample))
            index = str(np.argmax(predictions, axis=1))
            index = index.replace('[','')
            index = index.replace(']','')
            print("gesto:",prefix[int(index)])
    #return index
def prediction0(smp):
    #from tensorflow.keras.models import load_model
    #import pandas as pd
    #model = load_model('model_3')
    #print(mdl)
    #print(smp)
    predictions = model.predict(pd.DataFrame(smp))
    #print(predictions)
    #print(np.argmax(predictions, axis=1))
    #print(prefix[int(np.argmax(predictions, axis=1))])
    index = str(np.argmax(predictions, axis=1))
    index = index.replace('[','')
    index = index.replace(']','')
    print("gesto:",prefix[int(index)])
    #return index
#p0 = Process(target=prediction, args=())
#p0.start()
while True:
    line = ''
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:

        #file_a = open("a.txt","a")
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

            mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)
        a = array[9][0]
        b = array[9][1]
        c2 = math.pow(a,2) + math.pow(b,2)
        c = math.sqrt(c2)
        try:
            alpha = math.degrees(math.acos(b/c)) # 0 -> 90 -> 180/180 -> 90 -> 0
            if(a > 0):
                angle = 360 - alpha
            else:
                angle = alpha
            ref_angle = 180 - angle
        except Exception as e:
            print("Erro: ",e)
        #print("angle:",angle)
        aux_x = 200
        aux_y = h - 10
        ref_angle = math.radians(ref_angle)
        ref_scale = c / 200
        for i in range(1,21):
            x = array[i][0] / ref_scale
            y = array[i][1] / ref_scale
            x2 = (x*math.cos(ref_angle)) - (y*math.sin(ref_angle))
            y2 = (x*math.sin(ref_angle)) + (y*math.cos(ref_angle))
            array[i][0] = x2
            array[i][1] = y2
            sample[0].append(array[i][0])
            sample[0].append(array[i][1])
            sample[0].append(array[i][2])
        sample_array.append(sample)
        #print(sample)
        #sample_array[0] = sample
        print("main: ",len(sample_array))
        if __name__ == '__main__':
            with Manager() as manager:
                processes = [Process(target=prediction, args=(sample_array,)) for i in range(1)]
                # start all processes
                for process in processes:
                    process.start()
                #p1 = Process(target=prediction, args=(sample_array[]))
            #    p1.start()
            #p = Pool(processes = 4)
            #res = p.apply_async(prediction, (sample,))
            #print(res.get(timeout = 10))
            #print(p0.is_alive)
            #if last_p == 0:
            #    p1 = Process(target=prediction, args=(model,sample,))
            #    p1.start()
            #    last_p = 1
            #    print("starting 1")
            #else:
            #    p0 = Process(target=prediction, args=(model,sample,))
            #    p0.start()
            #    last_p = 0
            #    print("starting 0")
            #p0 = Process(target=prediction, args=(sample,))
            #p0.start()
            #prediction(sample)
            #time.sleep(10)
            #p0.terminate()
            #p.join()
        array = []
        sample[0] = []
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
