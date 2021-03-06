from pyimagesearch.centroidtracker_V2 import CentroidTracker
#import pyimagesearch.centroidtracker as centroidT
import time
import copy
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import dlib
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/video.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.5, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')

def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    count_mat = 0
    totalFrames = 0
    flagBlocoNotas = 1
    frameCru = 0
    while True:
        #if totalFrames > 0:
        #    frameAnterior = frameCru
        #if totalFrames == 0: # Pula o primeiro frame
        #    return_value, frame = vid.read()    
        return_value, frame = vid.read()

        #frameCru = frame

        #if totalFrames > 0:
        #    print(np.sum(frameCru - frameAnterior))



        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
    
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]
        
        ## IGUAL TCC
        rects = []
        confRects = []
        
        # Inicializa a nova variavel de rastreador de objetos
        trackers = []
        ###################### DETECCAO YOLO V4 ######################## inicio
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=500,
            max_total_size=500,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        #print(pred_bbox[0][:])
        #print(len(pred_bbox[0]))
        #print(pred_bbox[1][:])
        #print(len(pred_bbox[1]))
        #print(pred_bbox[2][:])
        #print(len(pred_bbox[2]))
        #print(pred_bbox[3])
        #print()
        image_h, image_w, _ = frame.shape
        for i in range(pred_bbox[3][0]):
            coor = pred_bbox[0][0][i]
            coor[0] = int(coor[0] * image_h)
            coor[2] = int(coor[2] * image_h)
            coor[1] = int(coor[1] * image_w)
            coor[3] = int(coor[3] * image_w)
            pred_bbox[0][0][i] = coor
            #print(coor)
            #print("scores: ", pred_bbox[1][0][i])
        ###################### DETECCAO YOLO V4 ######################## fim
        for i in range(pred_bbox[3][0]):
            # extrai a confian??a da predi????o
            confidence = pred_bbox[1][0][i]

            # filtra predi????es com baixa confian??a  ### TIRAR ISSO AQUI E DEICAR O centroidtracker_V2.py fazer isso
            if confidence > confidence_filter:
                #extrai as coordenadas das caixas delimitadoras
                box = pred_bbox[0][0][i]
                conf = pred_bbox[1][0][i]
                #(startX, startY, endX, endY) = box
                (startY, startX, endY, endX) = box # feito dessa forma para dar certo

                rects.append((startX, startY, endX, endY))
                confRects.append(conf)

        #for i in rects:
        #    print(i)

        #utiliza as BB feitas por mim no CVAT no primeiro frame
        if totalFrames >= 540:
            if flagBlocoNotas == 1:
                print("deregisterAll")
                ct.deregisterAll()
                flagBlocoNotas = 0
                rects = []
                confRects = []
                image_h, image_w, _ = frame.shape
                #for i in range(pred_bbox[3][0]):
                #    coor = pred_bbox[0][0][i]
                #    coor[0] = 0
                #    coor[2] = 0
                #    coor[1] = 0
                #    coor[3] = 0
                #    pred_bbox[0][0][i] = coor
                #print("bloco de notas .......................... comeco")
                #pathBlocoDeNotas = "./anotar_frames_video/Corrigido_CVAT/18-04-2021 001.txt"
                pathBlocoDeNotas = "./anotar_frames_video/Corrigido_CVAT/18-04-2021 - frame540.txt"
                with open(pathBlocoDeNotas) as f:
                    i = 0
                    for line in f:
                        
                        classe = int(line[0])
                        cX = float(line[2:10])
                        cY = float(line[11:19])
                        lX = float(line[20:28])
                        lY = float(line[29:37])

                        startX = int((cX-lX/2) * image_w)
                        startY = int((cY-lY/2) * image_h)
                        endX = int((cX+lX/2)*image_w)
                        endY = int((cY+lY/2)*image_h)

                        coor = pred_bbox[0][0][i]
                        coor[0] = startY
                        coor[1] = startX
                        coor[2] = endY
                        coor[3] = endX
                        #print(coor)
                        pred_bbox[0][0][i] = coor
                        pred_bbox[1][0][i] = 1

                        i += 1
                        rects.append((startX, startY, endX, endY))
                        confRects.append(1)
                    pred_bbox[3][0] = i

                    #for i in range(pred_bbox[3][0]):
                    #    print(rects[i])
                #print("bloco de notas .......................... fim")
            

        # atualiza os objetos do algoritmo de rastreamento de centroides
        objects = ct.update(rects, confRects, rgb)
        colors = ct.color
        desap = ct.disappeared
        BoundinBoxCt = ct.boundingB
        totalFrames += 1
        print("Frame atual:", totalFrames)

        fps = 1.0 / (time.time() - start_time)

        #image = utils.draw_bbox(frame, pred_bbox, show_BB=False)
        image = utils.draw_bbox(frame, pred_bbox, show_BB=True)
        image = utils.draw_bbox_tracker(image, objects, rects, colors, desap, BoundinBoxCt, totalObjetos = (ct.nextObjectID-1), fps = fps, frame = totalFrames)
        #image = utils.draw_frame(image, totalFrames)
        #image = utils.draw_bbox_neighbor(image, ct)        
        
        image_h, image_w, _ = frame.shape
        centroides = ct.objects
        
        
        #frameString = str(totalFrames + 1).zfill(3)
        frameString = str(totalFrames -1).zfill(3)
        print("frameString: ", frameString)
        
        #blocoDeNotaYolo = "./anotar_frames_video/Algoritmo/18-04-2021 " + frameString + ".txt"
        #18-04-202 - frame000
        blocoDeNotaYolo = "./anotar_frames_video/Algoritmo/18-04-2021 - frame" + frameString + ".txt"
        with open(blocoDeNotaYolo, 'w') as f:
            for (objectID, boundinBoxs) in BoundinBoxCt.items():
                #(startY, startX, endY, endX) = boundinBoxs
                (startX, startY, endX, endY) = boundinBoxs
                (cx, cy) = centroides[objectID]

                cXX = float(cx/image_w)
                cYY = float(cy/image_h)
                lX = float(2*(cXX - startX/image_w))
                lY = float(2*(cYY - startY/image_h))
                #lX = float(2*(endX/image_h - cXX))
                #lY = float(2*(endY/image_w - cYY))
                classe = 0
                
                cXX = str(format(cXX, '.6f'))
                cYY = str(format(cYY, '.6f'))
                lX = str(format(lX, '.6f'))
                lY = str(format(lY, '.6f'))
                classe = str(classe)

                #print()
                #print("cXX: ", cXX)
                #print("cYY: ", cYY)
                #print("lX: ", lX)
                #print("lY: ", lY)

                yoloNote = classe + ' ' + cXX + ' ' + cYY + ' ' + lX + ' ' + lY
                f.write(yoloNote)
                f.write('\n')
                #classe = int(line[0])
                #cY = float(line[2:10])
                #cX = float(line[11:19])
                #lY = float(line[20:28])
                #lX = float(line[29:37])

                #startY = int((cY-lY/2) * image_w)
                #startX = int((cX-lX/2) * image_h)
                #endY = int((cY+lY/2)*image_w)
                #endX = int((cX+lX/2)*image_h)

            
        
        ##### como ter todas as posi????es das BB detectadas pela YoLo ####

            
        count_mat = count_mat + 1
        if count_mat > 0:
            print("FPS: %.2f" % fps)
            print("Numero Laranjas: %i" % int(ct.nextObjectID-1))
            count_mat = 0
        result = np.asarray(image)
        #cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        #if not FLAGS.dont_show:
            #cv2.imshow("result", result)
        
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    #cv2.destroyAllWindows()

if __name__ == '__main__':
    
    ct = CentroidTracker(maxDisappeared=30, maxDistance=70, confiancaPrimeira = 0.85,
                         flagInputGreater=False, flagVelocitMoment = False,
                         flagTracker = True, flagBeirada = True)
    #ct2 = copy.deepcopy(ct)
    trackers = []
    skip_frames = 2
    confidence_filter = 0.75
    try:
        app.run(main)
    except SystemExit:
        pass
