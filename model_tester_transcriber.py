import cv2
import mediapipe as mp
import time
import numpy as np
import math
from tensorflow.keras.models import load_model
import pandas as pd
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QGridLayout,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QLabel,
)

WINDOW_SIZE = 250
DISPLAY_HEIGHT = 35
BUTTON_SIZE = 40

def gestureReader(interfaceWindow):
    cap = cv2.VideoCapture(0)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=2,model_complexity=1,min_detection_confidence=0.9,min_tracking_confidence=0.9)
    mpDraw = mp.solutions.drawing_utils
    cTime = 0
    pTime = 0# Function Start
    array = []
    sample = []
    sample.append([])
    gesture = 0
    phrase = ''
    cont = 0
    time.sleep(1)
    prefix = ["a","b","c","d","e","f","g","i","l","m","n","o","p","q","r","s","t","u","v","w","x","y","ponto","virgula","espaco","apagar"]
    prefix_char = ["a","b","c","d","e","f","g","i","l","m","n","o","p","q","r","s","t","u","v","w","x","y",".",",","_",""]
    last_index = ''
    index_count = 0
    model = load_model('model_4')
    test = 1
    while True:
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
            #print("c:",c)
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
            #print("angle:",angle)
            aux_x = 200
            aux_y = h - 10
            ref_angle = math.radians(ref_angle)
            ref_scale = c / 200
            #cv2.circle(img, (200, aux_y), 5, (139, 0, 0), cv2.FILLED)
            #cv2.circle(img, (350, aux_y), 5, (139, 0, 0), cv2.FILLED)
            for i in range(1,21):
                x = array[i][0] / ref_scale
                y = array[i][1] / ref_scale
                x2 = (x*math.cos(ref_angle)) - (y*math.sin(ref_angle))
                y2 = (x*math.sin(ref_angle)) + (y*math.cos(ref_angle))
                array[i][0] = x2
                array[i][1] = y2
                #print(array[i])
                #cv2.circle(img, (int(aux_x + array[i][0]), int(aux_y + array[i][1])), 5, (139, 0, 0), cv2.FILLED)
                #cv2.circle(img, (int(350-array[i][2]), int(aux_y + y)), 5, (139, 0, 0), cv2.FILLED)
                sample[0].append(array[i][0])
                sample[0].append(array[i][1])
                sample[0].append(array[i][2])
            predictions = model.predict(pd.DataFrame(sample))
            #print(predictions)
            #print(np.argmax(predictions, axis=1))
            #print(prefix[int(np.argmax(predictions, axis=1))])
            index = str(np.argmax(predictions, axis=1))
            index = index.replace('[','')
            index = index.replace(']','')
            print("gesto:",prefix[int(index)])
            interfaceWindow.label.setPixmap(QPixmap("images/"+prefix[int(index)]))
            if index == last_index:
                index_count += 1
                if index_count >= 20:
                    if index == "25":
                        phrase = phrase[:-1]
                    else:
                        phrase += prefix_char[int(index)]
                    index_count = 0
            last_index = index
            print("frase:",phrase)
            interfaceWindow.display.setText(phrase)
            array = []
            sample[0] = []
        # Time and FPS Calculation
        #print(results.multi_hand_landmarks.landmark[0][0])
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 3, (139,0,0), 3)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        image = QImage(img, img.shape[1], img.shape[0], img.strides[0], QImage.Format.Format_RGB888)
        interfaceWindow.video.setPixmap(QPixmap.fromImage(image))
        if(test):
            cv2.imshow("Image", img)
        cv2.waitKey(1)
        test = 0

class InterfaceWindow(QMainWindow):
    """PyCalc's main window (GUI or view)."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Transcritor LIBRAS")
        self.setFixedSize(850, 480)
        self.generalLayout = QHBoxLayout()
        self.leftWidget = QVBoxLayout()
        centralWidget = QWidget(self)
        centralWidget.setLayout(self.generalLayout)
        self.generalLayout.addLayout(self.leftWidget)
        self.setCentralWidget(centralWidget)
        self._createDisplay()
        #self._createButtons()
        self._createLabel()
        self._createVideo()
        #label = QLabel(self)
        #label.move(35, 10)
        #label.resize(175, 175)
        #pixmap = QPixmap("a.jpg")
        #label.setPixmap(pixmap)
        #self.generalLayout.addWidget(self.label)

    def _createDisplay(self):
        self.label_texto = QLabel("Texto transcrito:")
        self.label_texto.move(0,0)
        self.display = QLineEdit()
        self.display.setFixedHeight(DISPLAY_HEIGHT)
        self.display.move(0,0)
        self.display.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.display.setReadOnly(True)
        #self.display.setText("asdasd")
        self.leftWidget.addWidget(self.label_texto)
        self.leftWidget.addWidget(self.display)

    def _createLabel(self):
        # create a label
        self.label = QLabel()
        self.label.move(35, 10)
        self.label.resize(175, 175)
        self.label.setPixmap(QPixmap(""))
        self.leftWidget.addWidget(self.label)
        #th = Thread(self)
        #th.changePixmap.connect(self.setImage)
        #th.start()

    def _createVideo(self):
        # create a video
        self.video = QLabel()
        #self.video.move(35, 10)
        self.video.resize(480, 640)
        self.video.setPixmap(QPixmap(""))
        self.generalLayout.addWidget(self.video)

    def _createButtons(self):
        self.buttonMap = {}
        buttonsLayout = QGridLayout()
        keyBoard = [
            ["7", "8", "9", "/", "C"],
            ["4", "5", "6", "*", "("],
            ["1", "2", "3", "-", ")"],
            ["0", "00", ".", "+", "="],
        ]

        for row, keys in enumerate(keyBoard):
            for col, key in enumerate(keys):
                self.buttonMap[key] = QPushButton(key)
                self.buttonMap[key].setFixedSize(BUTTON_SIZE, BUTTON_SIZE)
                buttonsLayout.addWidget(self.buttonMap[key], row, col)

        self.generalLayout.addLayout(buttonsLayout)



    def setDisplayText(self, text):
        """Set the display's text."""
        self.display.setText(text)
        self.display.setFocus()

    def displayText(self):
        """Get the display's text."""
        return self.display.text()

    def clearDisplay(self):
        """Clear the display."""
        self.setDisplayText("")

def main():
    """PyCalc's main function."""
    interfaceApp = QApplication([])
    interfaceWindow = InterfaceWindow()
    interfaceWindow.show()
    #interfaceWindow.display.setText("qweqwe")
    #interfaceWindow.display.setText("zxczxc")
    gestureReader(interfaceWindow)
    sys.exit(interfaceApp.exec())

if __name__ == "__main__":
    main()
#https://en.wikipedia.org/wiki/Rotation_matrix
#https://www.geeksforgeeks.org/2d-transformation-in-computer-graphics-set-1-scaling-of-objects/
#https://www.aranacorp.com/en/displaying-an-opencv-image-in-a-pyqt-interface/
#https://stackoverflow.com/questions/57204782/show-an-opencv-image-with-pyqt5
