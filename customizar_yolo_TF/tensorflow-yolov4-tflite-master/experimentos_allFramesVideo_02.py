from absl import app, flags, logging
from absl.flags import FLAGS
from PIL import Image
import cv2
import numpy as np


#flags.DEFINE_string('video', './data/video/video.mp4', 'path to input video or set to 0 for webcam')
#flags.DEFINE_string('image', './data/video/video.mp4', 'path to input video or set to 0 for webcam')

#video_path = FLAGS.video
#image_path = FLAGS.image

video_path = "D:/ZZ Desktop/ZZ Mestrado/Converter YoLo to Tensor Flow/Sucesso_customizar_codigo_github/customization_YoLo_TF_2/customizar_yolo_TF/tensorflow-yolov4-tflite-master/anotar_frames_video/Video/18-04-2021.mp4"
image_path = "D:/ZZ Desktop/ZZ Mestrado/Converter YoLo to Tensor Flow/Sucesso_customizar_codigo_github/customization_YoLo_TF_2/customizar_yolo_TF/tensorflow-yolov4-tflite-master/anotar_frames_video/Frames/18-04-2021 001.jpg"

try:
    vid = cv2.VideoCapture(int(video_path))
except:
    vid = cv2.VideoCapture(video_path)

#img = cv2.imread('dumb.jpg', cv2.IMREAD_GRAYSCALE)

#img = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
img = cv2.imread(image_path)
cv2.imshow("imagem", img)
cv2.waitKey(5000)
#print(img)
#print(vid)
#print()
#print("AAAAAAAAAAAAAAAAAAAA")
#print()
return_value, frame = vid.read()
return_value, frame = vid.read()
return_value, frame = vid.read()
#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
cv2.imshow("imagem", frame)
cv2.waitKey(5000)

print(np.sum(frame-img))
print(frame-img)
#return_value, frame = vid.read()
#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#print(frame)

'''
while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            #print(len(frame))
            #print(np.size(frame))
            #print(frame.shape)
            print(np.sum(frame-img))
            #print(img.shape)
            #print(np.sum(img-img))
        else:
            print('Video has ended or failed, try a different video format!')
            break
'''