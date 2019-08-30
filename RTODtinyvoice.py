#less accurate but runs faster

import cv2 as cv
import numpy as np
import sys
import pyttsx3

confThreshold = 0.25
nmsThreshold = 0.40
inpWidth = 416
inpHeight = 416

classesFile = "coco.names"
classes = None

with open(classesFile,'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

modelConf = 'yolov3-tiny.cfg'
modelWeights = 'yolov3-tiny.weights'

def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIDs = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            
            scores = detection [5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > confThreshold:
                centerX = int(detection[0] * frameWidth)
                centerY = int(detection[1] * frameHeight)

                width = int(detection[2]* frameWidth)
                height = int(detection[3]*frameHeight )

                left = int(centerX - width/2)
                top = int(centerY - height/2)

                classIDs.append(classID)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes (boxes,confidences, confThreshold, nmsThreshold )

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        
        drawPred(classIDs[i], confidences[i], left, top, left + width, top + height)

def drawPred(classId, conf, left, top, right, bottom):
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    label = '%.2f' % conf

    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
        print(label)
        splitlabel = label.split(":")
        engine = pyttsx3.init()

        """ RATE"""
        rate = engine.getProperty('rate')
        engine.setProperty('rate', 180)


        """VOLUME"""
        volume = engine.getProperty('volume')
        engine.setProperty('volume', 1.0)

        """VOICE"""
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id) # 0 for male

        engine.say(splitlabel[0])
        engine.runAndWait()
        engine.stop()
        if splitlabel[0] == "bottle":
            intconf = float(splitlabel[1])
            if intconf > 0.30:
                print("terminating")
                sys.exit()
        
    cv.putText(frame, label, (left,top), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

def getOutputsNames(net):
    layersNames = net.getLayerNames()
   
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]



net = cv.dnn.readNetFromDarknet(modelConf, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

winName = 'DL OD with OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
cv.resizeWindow(winName, 1000,1000)

cap = cv.VideoCapture(0)

while cv.waitKey(1) < 0:

    hasFrame, frame = cap.read()
    
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop = False)

    net.setInput(blob)
    outs = net.forward (getOutputsNames(net))

    postprocess (frame, outs)

    cv.imshow(winName, frame)
