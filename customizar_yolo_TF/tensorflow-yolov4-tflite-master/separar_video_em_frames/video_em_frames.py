import cv2
nomeVideo = '18-04-2021.mp4'
vidcap = cv2.VideoCapture(nomeVideo)
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  frameBloco = str(count).zfill(3)
  nomeDoFrame = "./frames/" + nomeVideo[:-5] + " - frame" + frameBloco + ".jpg"

  print("..............")
  print("count: ", count)
  print("antes imwrite")
  #cv2.imwrite(nomeDoFrame, image)     # save frame as JPEG file
  print("depois imwrite")
  print("..............")
  print()
  
  if cv2.waitKey(10) == 27:                     # exit if Escape is hit
      break
  count += 1