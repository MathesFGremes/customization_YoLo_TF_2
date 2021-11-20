# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import random
import dlib
import cv2


trackerType = {
    "csrt": cv2.TrackerCSRT_create(),
	"kcf": cv2.TrackerKCF_create(),
    "boosting": cv2.TrackerBoosting_create(),
	"mil": cv2.TrackerMIL_create(),
    "tld": cv2.TrackerTLD_create(),
	"medianflow": cv2.TrackerMedianFlow_create(),
	"mosse": cv2.TrackerMOSSE_create()
}

class CentroidTracker:
	def __init__(self, maxDisappeared=50, maxDistance=50, confiancaPrimeira = 0.9, flagVelocitMoment = True, flagTracker = False, flagInputGreater = True, flagBeirada = True, trackingType = 'Dlib'):
		# initialize the next unique object ID along with two ordered
		# dictionaries used to keep track of mapping a given object
		# ID to its centroid and number of consecutive frames it has
		# been marked as "disappeared", respectively
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.confidence = OrderedDict()
		self.boundingB = OrderedDict()
		self.neighbor = OrderedDict()
		self.color = OrderedDict()
		self.relativeV = OrderedDict()
		self.arrayRelativeV = OrderedDict()
		self.trackerDLIB = OrderedDict()
		self.disappeared = OrderedDict()
		
		self.frameAtual = []
		self.frameAnterior = []
		self.percentBeirada = 0.02
		self.iouNewRegister = 0.15
		self.dMaxNeighbor = 3 # distancia maxima relativa em Bounding Box
		self.confiancaPrimeira = confiancaPrimeira
		self.trackingType = trackingType

		self.averageS = 0
		self.flagTracker = flagTracker
		self.flagVelocitMoment = flagVelocitMoment
		self.flagInputGreater = flagInputGreater
		self.flagBeirada = flagBeirada

		# store the number of maximum consecutive frames a given
		# object is allowed to be marked as "disappeared" until we
		# need to deregister the object from tracking
		self.maxDisappeared = maxDisappeared

		# store the maximum distance between centroids to associate
		# an object -- if the distance is larger than this maximum
		# distance we'll start to mark the object as "disappeared"
		self.maxDistance = maxDistance

	def register(self, centroid, boundingBox, confianca):
		# when registering an object we use the next available object
		# ID to store the centroid
		if confianca >= self.confiancaPrimeira:
			if self.flagBeirada:
				beirada = int(self.image_wy*(self.percentBeirada))
				beiradaInicio = beirada
				beiradaFim = self.image_wy - beirada
				cy = centroid[0]
				if (cy > beiradaInicio) and (cy < beiradaFim):
					self.objects[self.nextObjectID] = centroid
					self.color[self.nextObjectID] = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
					self.boundingB[self.nextObjectID] = boundingBox
					self.disappeared[self.nextObjectID] = 0
					self.relativeV[self.nextObjectID] = []
					self.confidence[self.nextObjectID] = confianca
					
					instantes = 0
					arrayV = [0 for x in range(60)]
					#arrayV = np.empty(60)
					#arrayV[:] = np.NaN
					averageIndividual = [0, 0]
					self.arrayRelativeV[self.nextObjectID] = {
						'instantes' : instantes,
						'array' : arrayV,
						'averageIndividual' : averageIndividual
					}
					self.trackerDLIB[self.nextObjectID] = []
					self.neighbor[self.nextObjectID] = {
						'Right' : [],
						'Left' : [],
						'Top' : [],
						'Down' : []
					}
					self.nextObjectID += 1
			else:
				self.objects[self.nextObjectID] = centroid
				self.color[self.nextObjectID] = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
				self.boundingB[self.nextObjectID] = boundingBox
				self.disappeared[self.nextObjectID] = 0
				self.relativeV[self.nextObjectID] = []
				self.confidence[self.nextObjectID] = confianca

				instantes = 0
				arrayV = [0 for x in range(60)]
				#arrayV = np.empty(60)
				#arrayV[:] = np.NaN
				averageIndividual = [0, 0]
				self.arrayRelativeV[self.nextObjectID] = {
					'instantes' : instantes,
					'array' : arrayV,
					'averageIndividual' : averageIndividual
				}
				self.trackerDLIB[self.nextObjectID] = []
				self.neighbor[self.nextObjectID] = {
						'Right' : [],
						'Left' : [],
						'Top' : [],
						'Down' : []
					}
				self.nextObjectID += 1

	def deregisterAll(self):
		#for objectID in list(self.objects.keys()):
		keys = list({key for key in self.objects})
		for objectID in keys:
			print('ok')
			self.deregister(objectID)

	def deregister(self, objectID):
		# to deregister an object ID we delete the object ID from
		# both of our respective dictionaries
		del self.objects[objectID]
		del self.boundingB[objectID]
		del self.confidence[objectID]
		del self.disappeared[objectID]
		del self.color[objectID]
		del self.relativeV[objectID]
		del self.arrayRelativeV[objectID]
		del self.trackerDLIB[objectID]
		del self.neighbor[objectID]
	
	def registraVizinho(self, idMorador, idVizinho, posicao):
		
		self.neighbor[idMorador][posicao] = {
			'objectID' : idVizinho,
			'object' : self.objects[idVizinho],
			'boundingB' : self.boundingB[idVizinho],
			'color' : self.color[idVizinho],
			'dRelativa': []
		}
		if posicao == 'Right' or posicao == 'Left':
			self.neighbor[idMorador][posicao]['dRelativa'] = abs(self.objects[idVizinho][0] - self.objects[idMorador][0])
		else:
			if posicao == 'Top' or posicao == 'Down':
				self.neighbor[idMorador][posicao]['dRelativa'] = abs(self.objects[idVizinho][1] - self.objects[idMorador][1])

		#print('idMorador: ', idMorador)
		#print('posicao: ', posicao)
		#print('objectID: ', self.neighbor[idMorador][posicao]['objectID'])
		#print('object: ', self.neighbor[idMorador][posicao]['object'])
		#print('boundingB: ', self.neighbor[idMorador][posicao]['boundingB'])
		#print('color: ', self.neighbor[idMorador][posicao]['color'])
		#print('dRelativa: ', self.neighbor[idMorador][posicao]['dRelativa'])
		#print()
	
	def decideRegistraVizinho(self, idMorador, idVizinho, flagRight, flagLeft, flagTop, flagDown):
		vizinhanca = self.checaProximidade(self.boundingB[idMorador], self.objects[idMorador], self.objects[idVizinho])
		if vizinhanca == 1 or vizinhanca == 2:
			if vizinhanca == 2 and flagRight == 0:
				flagRight = 1
				#vizinho esta a direita do morador
				
				#se nao há nenhum vizinho ja registrado, registra esse
				if len(self.neighbor[idMorador]['Right']) == 0:
					self.registraVizinho(idMorador, idVizinho, 'Right')
				#se ja ha um vizinho registrado, compara para ver quem é o vizinho mais proximo, ou atualiza a posicao se for o mesmo vizinho
				else:
					distanciaVizinhoNovo = abs(self.objects[idVizinho][0] - self.objects[idMorador][0])
					distanciaVizinhoAntigo = self.neighbor[idMorador]['Right']['dRelativa']
					if (distanciaVizinhoNovo < distanciaVizinhoAntigo) or (self.neighbor[idMorador]['Right']['objectID'] == idVizinho):
						self.registraVizinho(idMorador, idVizinho, 'Right')
			else:
				if flagLeft == 0:
					flagLeft = 1
					#vizinho esta a esquerda do morador
					#se ja ha um vizinho registrado, compara para ver quem é o vizinho mais proximo, ou atualiza a posicao se for o mesmo vizinho
					if len(self.neighbor[idMorador]['Left']) == 0:
						self.registraVizinho(idMorador, idVizinho, 'Left')
					#se ja ha um vizinho registrado, compara para ver quem é o vizinho mais proximo
					else:
						distanciaVizinhoNovo = abs(self.objects[idVizinho][0] - self.objects[idMorador][0])
						distanciaVizinhoAntigo = self.neighbor[idMorador]['Left']['dRelativa']
						if (distanciaVizinhoNovo < distanciaVizinhoAntigo) or (self.neighbor[idMorador]['Left']['objectID'] == idVizinho):
							self.registraVizinho(idMorador, idVizinho, 'Left')
		
		if vizinhanca == 3 or vizinhanca == 4:
			if vizinhanca == 4 and flagDown == 0:
				flagDown = 1
				#vizinho esta a baixo do morador
				
				#se nao há nenhum vizinho ja registrado, registra esse
				if len(self.neighbor[idMorador]['Down']) == 0:
					self.registraVizinho(idMorador, idVizinho, 'Down')
				#se ja ha um vizinho registrado, compara para ver quem é o vizinho mais proximo, ou atualiza a posicao se for o mesmo vizinho
				else:
					distanciaVizinhoNovo = abs(self.objects[idVizinho][0] - self.objects[idMorador][0])
					distanciaVizinhoAntigo = self.neighbor[idMorador]['Down']['dRelativa']
					if (distanciaVizinhoNovo < distanciaVizinhoAntigo) or (self.neighbor[idMorador]['Down']['objectID'] == idVizinho):
						self.registraVizinho(idMorador, idVizinho, 'Down')
			else:
				if flagTop == 0:
					flagTop = 1
					#vizinho esta a cima do morador
					#se ja ha um vizinho registrado, compara para ver quem é o vizinho mais proximo, ou atualiza a posicao se for o mesmo vizinho
					if len(self.neighbor[idMorador]['Top']) == 0:
						self.registraVizinho(idMorador, idVizinho, 'Top')
					#se ja ha um vizinho registrado, compara para ver quem é o vizinho mais proximo
					else:
						distanciaVizinhoNovo = abs(self.objects[idVizinho][0] - self.objects[idMorador][0])
						distanciaVizinhoAntigo = self.neighbor[idMorador]['Top']['dRelativa']
						if (distanciaVizinhoNovo < distanciaVizinhoAntigo) or (self.neighbor[idMorador]['Top']['objectID'] == idVizinho):
							self.registraVizinho(idMorador, idVizinho, 'Top')
		
		return flagRight, flagLeft, flagTop, flagDown

	def checaProximidade(self, bbMorador, centroidMorador, centroidVizinho):
		# 5 3 6
		# 1   2
		# 8 4 7
		startX = bbMorador[0]
		startY = bbMorador[1]
		endX = bbMorador[2]
		endY = bbMorador[3]
		cxM = centroidMorador[0]
		cyM = centroidMorador[1]
		cxV = centroidVizinho[0]
		cyV = centroidVizinho[1]

		#(self.objects[idVizinho][0] - self.objects[idMorador][0]) > 0
		if cyV >= startY and cyV <= endY:
			if (cxV - cxM) > 0:
				return 2 #vizinho na direita
			else:
				return 1 #vizinho na esquerda
		
		if cxV >= startX and cxV <= endX:
			if (cyV - cyM) > 0:
				return 4 #vizinho em baixo
			else:
				return 3 #vizinho em cima
		
		if cxV >= startX and cyV >= endY: #canto inferior direito
			return 7
		if cxV >= startX and cyV <= endY: #canto superior direito
			return 6
		if cxV <= startX and cyV <= endY: #canto superior esquerdo
			return 5
		if cxV <= startX and cyV >= endY: #canto inferior esquerdo
			return 8
		
	def closeNeighbor(self):
		objectIDs = list(self.objects.keys())
		objectCentroids = list(self.objects.values())
		#calcula a distancia entre todos os centroides
		D = dist.cdist(np.array(objectCentroids), np.array(objectCentroids))
		#ordena os elementos das linhas do menor para o maior da esquerda para a direita
		argsort = D.argsort()
		
		for i in np.arange(len(objectIDs)):
			flagRight = 0
			flagLeft = 0
			flagTop = 0
			flagDown = 0
			idMorador = objectIDs[i]
			#checa se o morador nao esta desaparecido
			if self.disappeared[idMorador] == 0:
				#distancia maxima para aceitar um vizinho com relacao a bounding box do morador
				dMaxNeigh = abs(self.boundingB[idMorador][2]-self.boundingB[idMorador][0])*self.dMaxNeighbor
				#ignora a primeira posicao que seria zero pois compara a distancia do morador com ele mesmo
				for j in np.arange(len(objectIDs)-1)+1:
					#checa se o vizinho esta dentro do valor limite
					if D[i, argsort[i,j]] < dMaxNeigh:
						idVizinho = objectIDs[argsort[i,j]]
						#print("id objeto: ", objectIDs[i])
						#print("id vizinhos: ", objectIDs[argsort[i,j]])
						#print("D vizinho: ", D[i, argsort[i,j]])
						#print()
						#checa se o vizinho nao esta desaparecido
						if self.disappeared[idVizinho] == 0:
							#checa qual a vizinhanca do vizinho
							flagRight, flagLeft, flagTop, flagDown = self.decideRegistraVizinho(idMorador, idVizinho, flagRight, flagLeft, flagTop, flagDown)
					else:
						break

	def averageSpeed(self):
		keyVelocit = list({key for key in self.relativeV if (len(self.relativeV[key]) > 0)})
		
		average = 0
		#print()
		#print(average)
		for key in keyVelocit:
			####################################
			average += self.relativeV[key]
			####################################
			self.arrayRelativeV[key]['instantes'] = self.arrayRelativeV[key]['instantes'] + 1
			k = self.arrayRelativeV[key]['instantes']
			self.arrayRelativeV[key]['array'][1:-1] =  self.arrayRelativeV[key]['array'][0:-2]
			
			self.arrayRelativeV[key]['array'][0] = self.relativeV[key]

			vx = self.arrayRelativeV[key]['averageIndividual'][0]
			vy = self.arrayRelativeV[key]['averageIndividual'][1]
			self.arrayRelativeV[key]['averageIndividual'] = (self.arrayRelativeV[key]['array'][0] + [vx*(k-1), vy*(k-1)])/k

			#print(average)
		if len(keyVelocit) > 0:
			############################################
			self.averageS = average/len(keyVelocit)
			############################################
			#print(self.averageS)
	
	def momentLost(self):
		self.averageSpeed()
		
		keyDisappeared = list({key for key in self.disappeared if (self.disappeared[key] > 0)})
		for key in keyDisappeared:
			#print(key)
			#print(self.objects[key])
			###############self.objects[key] = (self.objects[key] + self.averageS).astype(int)
			if self.arrayRelativeV[key]['instantes'] >= 60:
				print("ID Laranja: ", key)
				print("averageIndividual: ", self.arrayRelativeV[key]['averageIndividual'])
				self.objects[key] = (self.objects[key] + self.arrayRelativeV[key]['averageIndividual']).astype(int)

				### soma em X
				#self.boundingB[key][0] = (self.boundingB[key][0] + self.averageS[0]).astype(int)
				#self.boundingB[key][2] = (self.boundingB[key][2] + self.averageS[0]).astype(int)
				### soma em Y
				#self.boundingB[key][1] = (self.boundingB[key][1] + self.averageS[1]).astype(int)
				#self.boundingB[key][3] = (self.boundingB[key][3] + self.averageS[1]).astype(int)
				###############box = ((self.boundingB[key][0] + self.averageS[0]).astype(int),
				###############	   (self.boundingB[key][1] + self.averageS[1]).astype(int),
				###############	   (self.boundingB[key][2] + self.averageS[0]).astype(int),
				###############	   (self.boundingB[key][3] + self.averageS[1]).astype(int))
				
				box = ((self.boundingB[key][0] + self.arrayRelativeV[key]['averageIndividual'][0]).astype(int),
					(self.boundingB[key][1] + self.arrayRelativeV[key]['averageIndividual'][1]).astype(int),
					(self.boundingB[key][2] + self.arrayRelativeV[key]['averageIndividual'][0]).astype(int),
					(self.boundingB[key][3] + self.arrayRelativeV[key]['averageIndividual'][1]).astype(int))
				self.boundingB[key] = box
				#print(self.objects[key])
				#print()
	
	def deletaTrackingBeirada(self):
		keyDisappeared = list({key for key in self.disappeared if (self.disappeared[key] > 0)})
		print("rastreando: ", len(keyDisappeared))
		#beirada = int(self.image_wy*(0.0162))
		
		beirada = int(self.image_wy*(self.percentBeirada))
		beiradaInicio = beirada
		beiradaFim = self.image_wy - beirada
		#print("image_wy: ", self.image_wy)
		#print("image_hx: ", self.image_hx)
		#print("beiradaInicio: ", beiradaInicio)
		#print("beiradaFim: ", beiradaFim)
		#print()
		for key in keyDisappeared:
			cx = self.objects[key][0]
			#print("key: ", key)
			#print("cy: ", cy)
			
			if (cx < beiradaInicio) or (cx > beiradaFim):
				self.deregister(key)

	def firstTracking(self, idC):
		if self.trackingType == 'Dlib':
			self.trackerDLIB[idC] = dlib.correlation_tracker()
			rect = dlib.rectangle(self.boundingB[idC][0], self.boundingB[idC][1],
								self.boundingB[idC][2], self.boundingB[idC][3])
			self.trackerDLIB[idC].start_track(self.frameAnterior, rect)
		else:
			self.trackerDLIB[idC] = trackerType[self.trackingType]()
			initBB = (self.boundingB[idC][0], 
						self.boundingB[idC][1],
						self.boundingB[idC][2] - self.boundingB[idC][0],
						self.boundingB[idC][3] - self.boundingB[idC][1])
			self.trackerDLIB[idC].init(self.frameAnterior, initBB)

	def utilizeTrackingDLIB(self):
		
		keyDisappeared = list({key for key in self.disappeared if (self.disappeared[key] == 1)})
		for key in keyDisappeared:
			self.firstTracking(key)

		keyDisappeared = list({key for key in self.disappeared if (self.disappeared[key] > 0)})
		for key in keyDisappeared:
			if self.trackingType == 'Dlib':
				sc = self.trackerDLIB[key].update(self.frameAtual)
				pos = self.trackerDLIB[key].get_position()

				startX = int(pos.left())
				startY = int(pos.top())
				endX = int(pos.right())
				endY = int(pos.bottom())

				cX = int((startX + endX) / 2.0)
				cY = int((startY + endY) / 2.0)

				self.objects[key] = (cX, cY)
				self.boundingB[key] = (startX, startY, endX, endY)
			else:

				(success, box) = self.trackerDLIB[key].update(self.frameAtual)
				# check to see if the tracking was a success
				if success:
					(x, y, w, h) = [int(v) for v in box]
					#cv2.rectangle(frame, (x, y), (x + w, y + h),
					#	(0, 255, 0), 2)
					startX = int(x)
					startY = int(y)
					endX = int(x + w)
					endY = int(y + h)
					
					cX = int((startX + endX) / 2.0)
					cY = int((startY + endY) / 2.0)

					self.objects[key] = (cX, cY)
					self.boundingB[key] = (startX, startY, endX, endY)
				else:
					print("********************************")
					print("********************************")
					print("Tracking Falhou")
					print("********************************")
					print("********************************")
		#for tracker in trackers:
        #        #atualiza as caixas delimitadoras do rastreador de objetos
        #        sc = tracker.update(rgb)
        #        pos = tracker.get_position()
		#
        #        #obtem a nova posição do objeto
        #        startX = int(pos.left())
        #        startY = int(pos.top())
        #        endX = int(pos.right())
        #        endY = int(pos.bottom())
		#
        #        #adiciona a caixa delimitadora para a lista
        #        rects.append((startX, startY, endX, endY))

	def bb_intersection_over_union(self, bbCandidato, bbRegistrado):
		# determine the (x, y)-coordinates of the intersection rectangle
		xA = max(bbCandidato[0], bbRegistrado[0])
		yA = max(bbCandidato[1], bbRegistrado[1])
		xB = min(bbCandidato[2], bbRegistrado[2])
		yB = min(bbCandidato[3], bbRegistrado[3])

		
		# compute the area of intersection rectangle
		interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
		# compute the area of both the prediction and ground-truth
		# rectangles
		boxAArea = (bbCandidato[2] - bbCandidato[0] + 1) * (bbCandidato[3] - bbCandidato[1] + 1)
		boxBArea = (bbRegistrado[2] - bbRegistrado[0] + 1) * (bbRegistrado[3] - bbRegistrado[1] + 1)
		# compute the intersection over union by taking the intersection
		# area and dividing it by the sum of prediction + ground-truth
		# areas - the interesection area
		iou = interArea / float(boxAArea + boxBArea - interArea)
		# return the intersection over union value
		return iou

	def registerIOU(self, col, inputCentroids, boundingBoxs, confianca):
		#inputCentroids[col], boundingBoxs[tuple(inputCentroids[col])], confianca[tuple(inputCentroids[col])]
		objectIDs = list(self.objects.keys())
		objectCentroids = list(self.objects.values())

		D = dist.cdist([inputCentroids[col]], np.array(objectCentroids))
		#ordena os elementos das linhas do menor para o maior da esquerda para a direita
		argsort = D.argsort()
		print(".................")
		print('argsort: ', argsort)
		tam = len(argsort[0])

		j = 4
		if tam < j:
			j = tam
		
		iou = -1
		for i in np.arange(j):
			id = objectIDs[argsort[0][i]]
			
			if iou == -1:
				iou = self.bb_intersection_over_union(boundingBoxs[tuple(inputCentroids[col])], self.boundingB[id])
				print("iou: ", iou)
			else:
				iou2 = self.bb_intersection_over_union(boundingBoxs[tuple(inputCentroids[col])], self.boundingB[id])
				print("iou2: ", iou2)
				if iou2 > iou:
					iou = iou2
			
			if iou > 0:
				bbA = boundingBoxs[tuple(inputCentroids[col])]
				bbB = self.boundingB[id]
				if(bbB[0]>=bbA[0])and(bbB[1]>=bbA[1])and(bbB[2]<=bbA[2])and(bbB[3]<=bbA[3]):
					iou = 1
					print("BB contida A")
				else:
					if(bbB[0]<=bbA[0])and(bbB[1]<=bbA[1])and(bbB[2]>=bbA[2])and(bbB[3]>=bbA[3]):
						iou = 1
						print("BB contida B")

		print(".................")
		if iou < self.iouNewRegister:
			self.register(inputCentroids[col], boundingBoxs[tuple(inputCentroids[col])], confianca[tuple(inputCentroids[col])])



	def update(self, rects, confs, frame = []):
		if len(frame) > 0:
			image_hx, image_wy, _ = frame.shape
			self.image_hx = image_hx
			self.image_wy = image_wy
		self.frameAnterior = self.frameAtual
		self.frameAtual = frame
		boundingBoxs = OrderedDict()
		confianca = OrderedDict()
		# check to see if the list of input bounding box rectangles
		# is empty
		if len(rects) == 0:
			# loop over any existing tracked objects and mark them
			# as disappeared
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1
				self.relativeV[objectID] = []
				self.trackerDLIB[objectID] = []
				self.confidence[objectID] = []

				# if we have reached a maximum number of consecutive
				# frames where a given object has been marked as
				# missing, deregister it
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)

			if (self.flagVelocitMoment)and(not(self.flagTracker)):
				self.momentLost()
			else: 
				if self.flagTracker:
					if self.trackingType == 'Dlib':
						self.utilizeTrackingDLIB()
			if self.flagBeirada:
				self.deletaTrackingBeirada()

			# return early as there are no centroids or tracking info
			# to update
			return self.objects

		# initialize an array of input centroids for the current frame
		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		# loop over the bounding box rectangles
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			# use the bounding box coordinates to derive the centroid
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)
			boundingBoxs[tuple(inputCentroids[i])] = (startX, startY, endX, endY)
			confianca[tuple(inputCentroids[i])] = confs[i]

		# if we are currently not tracking any objects take the input
		# centroids and register each of them
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i], boundingBoxs[tuple(inputCentroids[i])], confianca[tuple(inputCentroids[i])])

		# otherwise, are are currently tracking objects so we need to
		# try to match the input centroids to existing object
		# centroids
		else:
			# grab the set of object IDs and corresponding centroids
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			# compute the distance between each pair of object
			# centroids and input centroids, respectively -- our
			# goal will be to match an input centroid to an existing
			# object centroid
			D = dist.cdist(np.array(objectCentroids), inputCentroids)
			### print('distancia: ', D)

			# in order to perform this matching we must (1) find the
			# smallest value in each row and then (2) sort the row
			# indexes based on their minimum values so that the row
			# with the smallest value as at the *front* of the index
			# list
			rows = D.min(axis=1).argsort()

			# next, we perform a similar process on the columns by
			# finding the smallest value in each column and then
			# sorting using the previously computed row index list
			cols = D.argmin(axis=1)[rows]

			# in order to determine if we need to update, register,
			# or deregister an object we need to keep track of which
			# of the rows and column indexes we have already examined
			usedRows = set()
			usedCols = set()

			# loop over the combination of the (row, column) index
			# tuples
			for (row, col) in zip(rows, cols):
				# if we have already examined either the row or
				# column value before, ignore it
				if row in usedRows or col in usedCols:
					continue

				# if the distance between centroids is greater than
				# the maximum distance, do not associate the two
				# centroids to the same object
				### print('D[row, col]: ', D[row, col])
				if D[row, col] > self.maxDistance:
					continue

				# otherwise, grab the object ID for the current row,
				# set its new centroid, and reset the disappeared
				# counter
				objectID = objectIDs[row]
				self.relativeV[objectID] = inputCentroids[col] - self.objects[objectID]
				self.objects[objectID] = inputCentroids[col]
				self.boundingB[objectID] = boundingBoxs[tuple(inputCentroids[col])]
				self.confidence[objectID] = confianca[tuple(inputCentroids[col])]
				self.disappeared[objectID] = 0
				self.trackerDLIB[objectID] = []
				
				arrayV = [0 for x in range(60)]
				#arrayV = np.empty(60)
				#arrayV[:] = np.NaN
				self.arrayRelativeV[objectID]['instantes'] = 0
				self.arrayRelativeV[objectID]['array'] = arrayV
				self.arrayRelativeV[objectID]['averageIndividual'] = [0, 0]

				# indicate that we have examined each of the row and
				# column indexes, respectively
				usedRows.add(row)
				usedCols.add(col)

			# compute both the row and column index we have NOT yet
			# examined
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)
			
			if self.flagInputGreater: #################################### FICA LIGADO AQUI QUE EU FIZ UMA BAGUNÇA
				# in the event that the number of object centroids is
				# equal or greater than the number of input centroids
				# we need to check and see if some of these objects have
				# potentially disappeared
				if D.shape[0] >= D.shape[1]:
					# loop over the unused row indexes
					for row in unusedRows:
						# grab the object ID for the corresponding row
						# index and increment the disappeared counter
						objectID = objectIDs[row]
						self.disappeared[objectID] += 1
						self.relativeV[objectID] = []
						self.confidence[objectID] = []

						# check to see if the number of consecutive
						# frames the object has been marked "disappeared"
						# for warrants deregistering the object
						if self.disappeared[objectID] > self.maxDisappeared:
							self.deregister(objectID)

				# otherwise, if the number of input centroids is greater
				# than the number of existing object centroids we need to
				# register each new input centroid as a trackable object
				else:
					for col in unusedCols:
						#self.register(inputCentroids[col], boundingBoxs[tuple(inputCentroids[col])], confianca[tuple(inputCentroids[col])])
						self.registerIOU(col, inputCentroids, boundingBoxs, confianca)
			else:
				if D.shape[0] >= D.shape[1]:
					# loop over the unused row indexes
					for row in unusedRows:
						# grab the object ID for the corresponding row
						# index and increment the disappeared counter
						objectID = objectIDs[row]
						self.disappeared[objectID] += 1
						self.relativeV[objectID] = []
						self.confidence[objectID] = []

						# check to see if the number of consecutive
						# frames the object has been marked "disappeared"
						# for warrants deregistering the object
						if self.disappeared[objectID] > self.maxDisappeared:
							self.deregister(objectID)
				for col in unusedCols:
					#self.register(inputCentroids[col], boundingBoxs[tuple(inputCentroids[col])], confianca[tuple(inputCentroids[col])])
					self.registerIOU(col, inputCentroids, boundingBoxs, confianca)

		# comput the average velocit in objects that are desapered
		if (self.flagVelocitMoment)and(not(self.flagTracker)):
			self.momentLost()
		else: # utiliza track DLIB nos objetos desaparecidos
			if self.flagTracker:
				if self.trackingType == 'Dlib':
					self.utilizeTrackingDLIB()
		if self.flagBeirada:
			self.deletaTrackingBeirada()
		
		# define os vizinhos
		self.closeNeighbor()

		# return the set of trackable objects
		return self.objects

if __name__ == '__main__':
	rects = []

	
	ct = CentroidTracker(maxDisappeared=1, maxDistance=50, flagTracker = True)
	for i in np.arange(10):
		j = i*10
		rects.append((j, j, j+2, j+2))
	
	ct.update(rects)
	print("Centroids: ", ct.objects)
	print("BBox: ", ct.boundingB)
	print("disappered: ", ct.disappeared)
	print("velocidade relativa: ", ct.relativeV)
	print()

	rects= []
	for i in np.arange(5):
		j = i*10
		m = 7
		h = 3
		rects.append((j+m, j+m, j+2-h, j+2-h))
	
	ct.update(rects)
	print("Centroids: ", ct.objects)
	print("BBox: ", ct.boundingB)
	print("disappered: ", ct.disappeared)
	print("velocidade relativa: ", ct.relativeV)

	print()
	print("velocidade media", ct.averageS)

	rects= []
	for i in np.arange(5):
		j = i*10
		m = 15
		h = 19
		rects.append((j+m, j+m, j+2-h, j+2-h))
	
	ct.update(rects)
	print("Centroids: ", ct.objects)
	print("BBox: ", ct.boundingB)
	print("disappered: ", ct.disappeared)
	print("velocidade relativa: ", ct.relativeV)

	print()
	print("velocidade media", ct.averageS)

	#keyVelocit = list({key for key in ct.relativeV if (len(ct.relativeV[key]) > 0)})
	#print(keyVelocit)
	#print()
	#ct.momentLost()

	#for k in keyVelocit:
	#	print(ct.relativeV[k])
	
	#print()
	#print("averageS: ", ct.averageS)
	#ct.averageSpeed()
	#print("averageS: ", ct.averageS)
	