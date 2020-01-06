# USAGE
# python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
d_id = {}
ids = 0

# load the COCO class labels our YOLO model was trained on
args = {}
seen={}
not_seen={}
args['confidence'] = 0.5
args['threshold'] = 0.3

args["yolo"] = "yolo-coco"

labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture('videos/overpass.mp4')
writer = None
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1
count = 0
# loop over frames from the video file stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()
	count += 1
	print("new frame read")

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
									swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > args["confidence"]:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
							args["threshold"])


	# ensure at least one detection exists
	dist_i={}
	print("objects in this frame",len(idxs))
	if(len(d_id)==0):
		print("in")
		
		if len(idxs) > 0:
			for kj in range(50):
	    			dist_i[kj]=999999
			for i in idxs.flatten():
				(x, y) = (boxes[i][0], boxes[i][1]) 
				(w, h) = (boxes[i][2], boxes[i][3]) 
				c = (x+w//2, y+h//2)
				color = [int(c) for c in COLORS[classIDs[i]]]
				#cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
				text = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
				
				d_id[ids] = c
				#cv2.putText(frame, text+'  id= '+str(ids), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
				ids+=1
	bounded={}
	done_c={}
	
	for itemp in range(len(idxs)):
		bounded[itemp]=0			
	for itemp in range(len(d_id)):
		done_c[itemp]=0
		
	for obj_id, obj in d_id.items():
		seen[obj_id]=0
		
		l=[]
		cs=[]
		dms=[]
		if len(idxs) > 0:
			for i in idxs.flatten():
				(x, y) = (boxes[i][0], boxes[i][1]) 
				(w, h) = (boxes[i][2], boxes[i][3])
				c = (x+w//2, y+h//2)
				dms.append(((x,y),(w,h)))
				
				dist = ((c[0]-obj[0])**2+(c[1]-obj[1])**2)**0.5
				l.append(dist)
				cs.append(c)
		#print(min(l))
		
		if min(l)<20:
		
			if (bounded[l.index(min(l))]==0):			
				print(l.index(min(l)))
				not_seen[obj_id]=0
				seen[obj_id]=1
				bounded[l.index(min(l))]+=1
				d_id[obj_id]=cs[l.index(min(l))]
				
				#print("in")
				temp=dms[l.index(min(l))]
				(x, y) = (temp[0][0], temp[0][1]) 
				(w, h) = (temp[1][0], temp[1][1])
				color = [int(c) for c in COLORS[classIDs[i]]]

				#print(obj_id)
				cv2.putText(frame, text+'  id= '+str(obj_id), (x, y + 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
				cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
				
	for itemp in range(len(bounded)):
		if(bounded[itemp]==0):
			bounded[itemp]+=1
			ids+=1
			print(ids)
			not_seen[ids]=0
			seen[ids]=1
				
			temp=dms[itemp]
			d_id[ids]=cs[itemp]
			(x, y) = (temp[0][0], temp[0][1]) 
			(w, h) = (temp[1][0], temp[1][1])
			color = [int(c) for c in COLORS[classIDs[i]]]

			cv2.putText(frame, text+'  id= '+str(ids), (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
	print(bounded)
	print(d_id)
	print(not_seen)
	for obj_id, obj in d_id.items():
		if not seen[obj_id]:
			not_seen[obj_id]+=1
			print("incremented for",obj_id,not_seen[obj_id])
			if(not_seen[obj_id]==15):
				print("deleted")
				del seen[obj_id]
				del not_seen[obj_id]
				del d_id[obj_id]
	if writer is None:
		# initialize our video writer
		fourcc=cv2.VideoWriter_fourcc(*"MJPG")
		writer=cv2.VideoWriter("output4.avi", fourcc, 30,
									(frame.shape[1], frame.shape[0]), True)

		# some information on processing single frame
		if total > 0:
			elap=(end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))

	# write the output frame to disk
	# cv2.imshow('Frame',frame)

	# Press Q on keyboard to  exit
	writer.write(frame)
	if cv2.waitKey(25) & 0xFF == ord('q'):
		break


# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
