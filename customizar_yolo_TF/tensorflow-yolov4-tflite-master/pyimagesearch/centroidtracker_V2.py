# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import random
import dlib

class CentroidTracker:
	def __init__(self, maxDisappeared=50, maxDistance=50, flagVelocitMoment = True, flagTracker = False, flagInputGreater = True):
		# initialize the next unique object ID along with two ordered
		# dictionaries used to keep track of mapping a given object
		# ID to its centroid and number of consecutive frames it has
		# been marked as "disappeared", respectively
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.boundingB = OrderedDict()
		self.color = OrderedDict()
		self.relativeV = OrderedDict()
		self.trackerDLIB = OrderedDict()
		self.disappeared = OrderedDict()
		
		self.frameAtual = []
		self.frameAnterior = []

		self.averageS = 0
		self.flagTracker = flagTracker
		self.flagVelocitMoment = flagVelocitMoment
		self.flagInputGreater = flagInputGreater

		# store the number of maximum consecutive frames a given
		# object is allowed to be marked as "disappeared" until we
		# need to deregister the object from tracking
		self.maxDisappeared = maxDisappeared

		# store the maximum distance between centroids to associate
		# an object -- if the distance is larger than this maximum
		# distance we'll start to mark the object as "disappeared"
		self.maxDistance = maxDistance

	def register(self, centroid, boundingBox):
		# when registering an object we use the next available object
		# ID to store the centroid
		self.objects[self.nextObjectID] = centroid
		self.color[self.nextObjectID] = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
		self.boundingB[self.nextObjectID] = boundingBox
		self.disappeared[self.nextObjectID] = 0
		self.relativeV[self.nextObjectID] = []
		self.trackerDLIB[self.nextObjectID] = []
		self.nextObjectID += 1

	def deregister(self, objectID):
		# to deregister an object ID we delete the object ID from
		# both of our respective dictionaries
		del self.objects[objectID]
		del self.boundingB[objectID]
		del self.disappeared[objectID]
		del self.color[objectID]
		del self.relativeV[objectID]
		del self.trackerDLIB[objectID]

	def averageSpeed(self):
		keyVelocit = list({key for key in self.relativeV if (len(self.relativeV[key]) > 0)})
		
		average = 0
		#print()
		#print(average)
		for key in keyVelocit:
			average += self.relativeV[key]
			#print(average)
		if len(keyVelocit) > 0:
			self.averageS = average/len(keyVelocit)
			#print(self.averageS)
	
	def momentLost(self):
		self.averageSpeed()
		
		keyDisappeared = list({key for key in self.disappeared if (self.disappeared[key] > 0)})
		for key in keyDisappeared:
			#print(key)
			#print(self.objects[key])
			self.objects[key] = (self.objects[key] + self.averageS).astype(int)
			### soma em X
			#self.boundingB[key][0] = (self.boundingB[key][0] + self.averageS[0]).astype(int)
			#self.boundingB[key][2] = (self.boundingB[key][2] + self.averageS[0]).astype(int)
			### soma em Y
			#self.boundingB[key][1] = (self.boundingB[key][1] + self.averageS[1]).astype(int)
			#self.boundingB[key][3] = (self.boundingB[key][3] + self.averageS[1]).astype(int)
			box = ((self.boundingB[key][0] + self.averageS[0]).astype(int),
				   (self.boundingB[key][1] + self.averageS[1]).astype(int),
				   (self.boundingB[key][2] + self.averageS[0]).astype(int),
				   (self.boundingB[key][3] + self.averageS[1]).astype(int))
			self.boundingB[key] = box
			#print(self.objects[key])
			#print()
	
	def firstTracking(self, idC):
		self.trackerDLIB[idC] = dlib.correlation_tracker()
		rect = dlib.rectangle(self.boundingB[idC][0], self.boundingB[idC][1],
							  self.boundingB[idC][2], self.boundingB[idC][3])
		self.trackerDLIB[idC].start_track(self.frameAnterior, rect)

	def utilizeTrackingDLIB(self):
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
		keyDisappeared = list({key for key in self.disappeared if (self.disappeared[key] == 1)})
		for key in keyDisappeared:
			self.firstTracking(key)

		keyDisappeared = list({key for key in self.disappeared if (self.disappeared[key] > 0)})
		for key in keyDisappeared:
			sc = self.trackerDLIB.update(self.frameAtual)
			pos = self.trackerDLIB.get_position()

			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

	def update(self, rects, frame = []):
		self.frameAnterior = self.frameAtual
		self.frameAtual = frame
		boundingBoxs = OrderedDict()
		# check to see if the list of input bounding box rectangles
		# is empty
		if len(rects) == 0:
			# loop over any existing tracked objects and mark them
			# as disappeared
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1
				self.relativeV[objectID] = []

				# if we have reached a maximum number of consecutive
				# frames where a given object has been marked as
				# missing, deregister it
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)

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

		# if we are currently not tracking any objects take the input
		# centroids and register each of them
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i], boundingBoxs[tuple(inputCentroids[i])])

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
				self.disappeared[objectID] = 0

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
						self.register(inputCentroids[col], boundingBoxs[tuple(inputCentroids[col])])
			else:
				if D.shape[0] >= D.shape[1]:
					# loop over the unused row indexes
					for row in unusedRows:
						# grab the object ID for the corresponding row
						# index and increment the disappeared counter
						objectID = objectIDs[row]
						self.disappeared[objectID] += 1
						self.relativeV[objectID] = []

						# check to see if the number of consecutive
						# frames the object has been marked "disappeared"
						# for warrants deregistering the object
						if self.disappeared[objectID] > self.maxDisappeared:
							self.deregister(objectID)
				for col in unusedCols:
					self.register(inputCentroids[col], boundingBoxs[tuple(inputCentroids[col])])

		# comput the average velocit in objects that are desapered
		if self.flagVelocitMoment:
			self.momentLost()

		# return the set of trackable objects
		return self.objects

if __name__ == '__main__':
	rects = []
	ct = CentroidTracker(maxDisappeared=50, maxDistance=50)
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