def bb_intersection_over_union(bbCandidato, bbRegistrado):
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

bbNovo = (10, 10, 20, 20)
bbAntigo = (10, 10, 20, 20)

print(bb_intersection_over_union(bbNovo, bbAntigo))

