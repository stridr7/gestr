import cv2 as cv
import mediapipe as mp
import time as t
import pyautogui as pag
import numpy as np
import math

mapGestr = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mapGestr.Hands(max_num_hands = 1, min_detection_confidence = .65)
clocX = 0
plocX = 0
clocY = 0
plocY = 0
res_x = 640
res_y = 480
screenWidth, screenHeight = pag.size()
frmcapture = cv.VideoCapture(0)
frmcapture.set(3, res_x)
frmcapture.set(4, res_y)
while True:
    success, img = frmcapture.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
 
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            keypoints = []
            for land_mark in hand.landmark:

                img_h, img_w, img_c = img.shape
                
                cv.rectangle(img, (100, 100), (res_x - 100, res_y-100),
                (255, 0, 255), 2)
                
                keypoints.append({'X': land_mark.x, 'Y': land_mark.y})


            ipos_x, ipos_y = int(keypoints[8]['X'] * img_w), int(keypoints[8]['Y'] * img_h)
            cv.circle(img, (ipos_x, ipos_y), 15, (0,139,139), cv.FILLED)
            x = np.interp(ipos_x, (100, res_x-100) ,(0, screenWidth))
            y = np.interp(ipos_y, (100, res_y-100) ,(0, screenHeight))
            clocX = plocX + (x - plocX)/5
            clocY = plocY + (y - plocY)/5
            if clocX>0 and clocY>0:
                pag.moveTo(screenWidth-clocX, clocY)
            plocX, plocY = clocX, clocY

            mpos_x, mpos_y = int(keypoints[12]['X'] * img_w), int(keypoints[12]['Y'] * img_h)
            cv.circle(img, (mpos_x, mpos_y), 15, (0,128,128), cv.FILLED)
            length = math.hypot(ipos_x - mpos_x, ipos_y - mpos_y)
            if length<=25:
                pag.click()
