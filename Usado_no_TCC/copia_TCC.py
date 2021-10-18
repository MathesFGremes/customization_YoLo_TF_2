from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

args = {
    "input":""
    ,"output":""
    ,"confidence":""
    ,"skip_frames":""
}

vs = cv2.VideoCapture(args["input"])
writer = None
totalFrames = 0
W = None
H = None
vetor = []

# inicia o algoritmo de rastreamento de centroides
ct = CentroidTracker(maxDisappeared=33, maxDistance=25)
trackers = []


start_time = time.time()

fps = 0


#cicle pelos frames
while True:
    # Selecione o proximo frame do video
    frame = vs.read()
    frame = frame[1] if args.get("input", False) else frame

    # Se nenhum frame foi obitido, acabe o loop
    if args["input"] is not None and frame is None:
        break

    #redimensione o frame para 500 pixels
    #redimensione o frame para RGB, para utilizar Dlib
    frame = imutils.resize(frame, width=416)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # if the frame dimensions are empty, set them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    #Escreva o vídeo para uma saida
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)


    # Inicializa a variavel que ira receber as caixas delimitadora
    rects = []
    corBox = []

    # checa se ira utilizar o detector de objetos ou o rastreados
    if totalFrames % args["skip_frames"] == 0:
        # Inicializa a nova variavel de rastreador de objetos
        trackers = []
        corBox = (0, 0, 255)

        #aplica YoLo v3
        r_image, ObjectsList = yolo.detect_imag(frame)

        for i in np.arange(0, len(ObjectsList)):
            # extrai a confiança da predição
            confidence = float(objectsList[i][7])

            # filtra predições com baixa condiança
            if confidence > args["confidence"]:
                idx = ObjectsList[i][6].split()[0]
                #filtra objetos que não pertencem as seguintes classes
                if idx != "car" and idx != "bus" and idx != "truck":
                    continue

                #extrai as coordenadas das caixas delimitadoras
                box = (ObjectsList[i][1], ObjectsList[i][0], ObjectsList[i][3], ObjectsList[i][2])
                (startX, startY, endX, endY) = box


                rects.append((startX, startY, endX, endY))


                #aplica rastreador da biblioteca Dlib
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)

                #adiciona os rastreadores para a lisa dos rastreadores
                trackers.append(tracker)

    #se o detector de objetos não for ser utilizado, utilizara o
    #rastreador de objetos
    else:
        corBox = (255, 255, 0)
        for tracker in trackers:
            #atualiza as caixas delimitadoras do rastreador de objetos
            sc = tracker.update(rgb)
            pos = tracker.get_position()

            #obtem a nova posição do objeto
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            #adiciona a caixa delimitadora para a lista
            rects.append((startX, startY, endX, endY))

    # atualiza os objetos do algoritmo de rastreamento de centroides
    objects = ct.update(rects)

    for (objectID, centroid) in objects.items():
        #desenhe o Id do objeto e o seu centroide no frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.circle(frame, (centroid[0], centroid[1]), 2, (0, 255, 0), -1)

    # desenhe as caixas delimitadoras no frame
    for rect in rects:
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), corBox, 1)
    
    # desenhe o numero do frame
    cv2.putText(frame, str(totalFrames), (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    totalFrames += 1

    fps += 1
    TIME = time.time() - start_time
    print("FPS:", fps / TIME, "trackers", len(trackers))
    fps = 0
    start_time = time.time()

if writer is not None:
    writer.release()

if not args.get("input", False):
    vs.stop()
else:
    vs.release()

cv2.destroyAllWindows()


