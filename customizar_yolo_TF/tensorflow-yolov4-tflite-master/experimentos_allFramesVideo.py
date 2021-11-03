lista = []

with open("./customizar_yolo_TF/tensorflow-yolov4-tflite-master/anotar_frames_video\Corrigido_CVAT/18-04-2021 001.txt") as f:
    for line in f:
        #print()
        #print("......")
        #print(line)
        #print(len(line))
        #print(line[0])
        #print(line[2:10])
        #print(line[11:19])
        #print(line[20:28])
        #print(line[29:37])

        #(startY, startX, endY, endX)
        classe = int(line[0])
        startY = float(line[2:10])
        startX = float(line[11:19])
        endY = float(line[20:28])
        endX = float(line[29:37])
        lista.append((classe, startY, startX, endY, endX))
        #print(line[21:30])
        #print(line[22:38])

for (i, (classe ,startY, startX, endY, endX)) in enumerate(lista):
    print(classe ,startY, startX, endY, endX)
    classe2 = str(classe) 
    startY2 = str(startY*2) 
    startX2 = str(startX*2) 
    endY2 = str(endY*2) 
    endX2 = str(endX*2)

    if startY2[0] == '1':
        #startY2[0] = '0'
        startY2 = '0' + startY2[1:]
    
    if startX2[0] == '1':
        #startX2[0] = '0'
        startX2 = '0' + startX2[1:]
    
    if endY2[0] == '1':
        #endY2[0] = '0'
        endY2 = '0' + endY2[1:]
    
    if endX2[0] == '1':
        #endX2[0] = '0'
        endX2 = '0' + endX2[1:]

    #lista[i] = (str(classe) ,str(startY*2), str(startX*2), str(endY*2), str(endX*2))
    sY = 8 - len(startY2)
    sX = 8 - len(startX2)
    eY = 8 - len(endY2)
    eX = 8 - len(endX2)

    startY2 = startY2 + '0'*sY 
    startX2 = startX2 + '0'*sX 
    endY2 = endY2 + '0'*eY 
    endX2 = endX2 + '0'*eX

    lista[i] = (classe2 ,startY2, startX2, endY2, endX2)


print()
print("........................")
print()

for (i, (classe ,startY, startX, endY, endX)) in enumerate(lista):
    print(classe ,startY, startX, endY, endX)
    print(len(classe), len(startY), len(startX), len(endY), len(endX))
    print()
    
    #lista[i] = (classe ,startY*2, startX*2, endY*2, endX*2)


#lines = ['Readme', 'How to write text files in Python']
with open("./customizar_yolo_TF/tensorflow-yolov4-tflite-master/anotar_frames_video\Corrigido_CVAT/18-04-2021 002.txt", 'w') as f:
    #for line in lines:
    #    f.write(line)
    #    f.write('\n')
    for (i, (classe ,startY, startX, endY, endX)) in enumerate(lista):
    #print(classe ,startY, startX, endY, endX)
    #print(len(classe), len(startY), len(startX), len(endY), len(endX))
    #print()
        f.write(classe + ' ' + startY + ' ' + startX + ' ' + endY + ' ' + endX)
        f.write('\n')