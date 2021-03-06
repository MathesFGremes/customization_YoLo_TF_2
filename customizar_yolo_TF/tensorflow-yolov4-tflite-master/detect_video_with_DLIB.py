from pyimagesearch.centroidtracker import CentroidTracker
#import pyimagesearch.centroidtracker as centroidT
import time
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
    while True:
        return_value, frame = vid.read()
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
        flag_ambos = 1
        flag_detection = 0
        if (totalFrames % skip_frames == 0) or (flag_ambos == 1):
            flag_detection = 1
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
            image_h, image_w, _ = frame.shape
            for i in range(pred_bbox[3][0]):
                coor = pred_bbox[0][0][i]
                coor[0] = int(coor[0] * image_h)
                coor[2] = int(coor[2] * image_h)
                coor[1] = int(coor[1] * image_w)
                coor[3] = int(coor[3] * image_w)
                pred_bbox[0][0][i] = coor
            ###################### DETECCAO YOLO V4 ######################## fim
            for i in range(pred_bbox[3][0]):
                # extrai a confian??a da predi????o
                confidence = pred_bbox[1][0][i]

                # filtra predi????es com baixa condian??a
                if confidence > confidence_filter:
                    #extrai as coordenadas das caixas delimitadoras
                    box = pred_bbox[0][0][i]
                    #(startX, startY, endX, endY) = box
                    (startY, startX, endY, endX) = box

                    rects.append((startX, startY, endX, endY))
                    if flag_ambos == 0:
                        #aplica rastreador da biblioteca Dlib
                        tracker = dlib.correlation_tracker()
                        rect = dlib.rectangle(startX, startY, endX, endY)
                        tracker.start_track(rgb, rect)

                        #adiciona os rastreadores para a lisa dos rastreadores
                        trackers.append(tracker)
        #se o detector de objetos n??o for ser utilizado, utilizara o
        #rastreador de objetos
        #else:
        if (totalFrames % skip_frames != 0) or (flag_ambos == 0):
            #corBox = (255, 255, 0)
            for tracker in trackers:
                #atualiza as caixas delimitadoras do rastreador de objetos
                sc = tracker.update(rgb)
                pos = tracker.get_position()

                #obtem a nova posi????o do objeto
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                #adiciona a caixa delimitadora para a lista
                rects.append((startX, startY, endX, endY))
        # atualiza os objetos do algoritmo de rastreamento de centroides
        objects = ct.update(rects)
        colors = ct.color
        totalFrames += 1
        #if flag_detection == 1:
        #    image = utils.draw_bbox(frame, pred_bbox)
        #else:
        #    image = utils.draw_bbox_tracker(frame, objects, rects, colors)        
        image = utils.draw_bbox(frame, pred_bbox)
        image = utils.draw_bbox_tracker(image, objects, rects, colors)        
        
        
        
        
        #print(pred_bbox[0].size)
        #print(pred_bbox[1])
        #print(pred_bbox[2])
        #print(pred_bbox[3])

        ##### como ter todas as posi????es das BB detectadas pela YoLo ####
        
        '''
        image_h, image_w, _ = frame.shape
        for i in range(pred_bbox[3][0]):
            coor = pred_bbox[0][0][i]
            #coor[0] = int(coor[0] * image_h)
            #coor[2] = int(coor[2] * image_h)
            #coor[1] = int(coor[1] * image_w)
            #coor[3] = int(coor[3] * image_w)

            print(coor[0], coor[2], coor[1], coor[3])
            #print(pred_bbox[0][0][i], pred_bbox[1][0][i], pred_bbox[2][0][i])
        print(pred_bbox[3][0])
        print(image_h, image_w)
        print()
        '''
        ##### como ter todas as posi????es das BB detectadas pela YoLo ####

        fps = 1.0 / (time.time() - start_time)
            
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
    ct = CentroidTracker(maxDisappeared=10, maxDistance=25)
    trackers = []
    skip_frames = 2
    confidence_filter = 0.75
    try:
        app.run(main)
    except SystemExit:
        pass
