import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import cv2
import numpy as np
import warnings
import time
from torch.autograd import Variable
from PIL import Image
from data import BaseTransform
from ssd import build_ssd
from math import pi as pi
from random import random, randint
from math import cos, sin
#from CodeAES import encryption
#from CodeAES import encrypt, decrypt


warnings.filterwarnings("once")
THRES = 600

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='ssd300_WIDER_100455.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--encrypt', default=False, type=bool,
                    help='encryption on or off')
parser.add_argument('--optical_flow', default=False, type=bool,
                    help='use optical flow ')
parser.add_argument('--multi_tracking', default=True, type=bool,
                    help='use multi_tracking ')
parser.add_argument('--jetson', default=False, type=bool,
                    help='Running on Jetson')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def gstreamer_pipeline(capture_width=3280, capture_height=2464, display_width=820, display_height=616, framerate=21, flip_method=2):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (capture_width, capture_height, framerate, flip_method, display_width, display_height)
    )

def face_detect():
	start_time = time.time()
	ret, image = cap.read()

	# Process input
	[h, w] = image.shape[:2]
	image = cv2.flip(image, 1)
	x = torch.from_numpy(transformer(image)[0]).permute(2, 0, 1)
	x = Variable(x.unsqueeze(0))

	if args.cuda:
		x = x.cuda()

    # Generate Detection
	y = net(x)
	detections = y.data

	end_time = time.time()

    # Scale each detection back up to the image
	scale = torch.Tensor([image.shape[1], image.shape[0],
                     image.shape[1], image.shape[0]])

    # Go through the boxes and keep only those with conf > threshold
	boxes = []
	j = 0
	while detections[0, 1, j, 0] >= 0.35:
		pt = (detections[0, 1, j, 1:]*scale).cpu().numpy()
		x1, y1, x2, y2 = pt
		if x2 - x1 < THRES and y2 - y1 < THRES: # Filter out the biggest box
			boxes.append((pt[0], pt[1], pt[2], pt[3]))
		j += 1

    # Draw bounding boxes
	for box in boxes:
		x1, y1, x2, y2 = box
		image= cv2.rectangle(image, (x1,y1), (x2,y2), (0, 0, 255), 2)

    # Encrypt Image
	if args.encrypt:
		if encrypt_status:
			image, IV  = encryptor.encrypt(boxes, image)

	    # Encode Image


	    # Decrypt Image
		if decrypt_status:
			image = encryptor.decrypt(boxes, image, IV)

    # Display FPS
	fps = 1 / (end_time - start_time)
	image = cv2.putText(image, 'fps: %.3f' % fps, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))

	cv2.imshow("Face Detect", image)
	return image, boxes

def face_detect_every_frame():

	while cv2.getWindowProperty("Face Detect", 0) >= 0:

		keyCode = cv2.waitKey(30) & 0xFF

		face_detect()

	    # Process and update keycode
		if chr(keyCode) == 'q' or keyCode == 27:
			break
		if chr(keyCode) == 'e':
			encrypt_status = not encrypt_status
		if chr(keyCode) == 'd':
			decrypt_status = not decrypt_status
		if chr(keyCode) == 'v':
			verbose = not verbose

	cap.release()
	cv2.destroyAllWindows()

def generate_random_points(n, radius, center):

	points = []
	for i in range(n):
		t = 2*pi*random()
		r = randint(0, int(radius))
		points.append([[center[0] + r*cos(t), center[1] + r*sin(t)]])
	print(points)
	return points


def multi_tracking():
	old_frame, boxes = face_detect()
	old_frame_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
	multiTracker = cv2.MultiTracker_create()
	for box in boxes:
		multiTracker.add(cv2.TrackerMOSSE_create(), old_frame_gray, box)
	
	while cv2.getWindowProperty("Face Detect", 0) >= 0:
		start_time = time.time()
		ret, image = cap.read()
		image = cv2.flip(image, 1)
		frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		success, boxes = multiTracker.update(frame_gray)
		for i, newbox in enumerate(boxes):
			p1 = (int(newbox[0]), int(newbox[1]))
			p2 = (int(newbox[2]), int(newbox[3]))
			cv2.rectangle(image, p1, p2, randint(0,255), 2, 1)
 
		end_time = time.time()
 		# show frame
		fps = 1 / (end_time - start_time)

		image = cv2.putText(image, 'success: %r' % success, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
		image = cv2.putText(image, 'fps: %.3f' % fps, (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))

		cv2.imshow('Face Detect', image)
   
 
  		# quit on ESC button
		if cv2.waitKey(1) & 0xFF == 27:
			break



def optical_flow():

	old_frame, boxes = face_detect()
	old_frame_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

	all_points = []
	for box in boxes:
	# 	top_left = [[box[0], box[1]]]
	# 	bottom_right = [[box[2], box[3]]]
	# 	top_right = [[box[0] + (box[2] - box[0]), box[1]]]
	# 	bottom_left = [[box[2] - (box[2] - box[0]), box[3]]]
	# 	all_points.append(top_left)
	# 	all_points.append(bottom_right)
	# 	all_points.append(top_right)
	# 	all_points.append(bottom_left)

	# print(all_points)
	# p0 = np.float32(all_points)

		box_width = (box[2] - box[0])
		box_height = (box[3] - box[1])
		mid_box = [box[0] + box_width / 2, box[1] + box_height / 2]

		all_points.append([mid_box])
		all_points = all_points + generate_random_points(10, box_width/2, mid_box)
	
	print(all_points)
	p0 = np.float32(all_points)

	delay_start = time.time()

	while cv2.getWindowProperty("Face Detect", 0) >= 0:
		start_time = time.time()
		ret, image = cap.read()
		image = cv2.flip(image, 1)

		frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		keyCode = cv2.waitKey(30) & 0xFF

		# Optical Flow
		if (args.optical_flow):

			# Create some random colors
			color = np.random.randint(0,255,(100,3))

			# Create a mask image for drawing purposes
			mask = np.zeros_like(old_frame)

			# Parameters for lucas kanade optical flow
			lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

		    # calculate optical flow
			if p0.any()!=None:
				p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame_gray, frame_gray, p0, None, **lk_params)
				print('p1', p1)
				print('st', st)
				print('err', err)

			    # Select good points
				good_new = p1[st==1]
				good_old = p0[st==1]

				x_pt = ()
				y_pt = ()

			    # draw the tracks
				for i,(new,old) in enumerate(zip(good_new,good_old)):
					a,b = new.ravel()
					c,d = old.ravel()
					mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
					frame = cv2.circle(image,(a,b),5,color[i].tolist(),-1)
					# if i%4==0:
					# 	x_pt = (a,b)
					# elif i%2!=0 and i%3!=0:
					# 	y_pt = (a,b)
					# 	print(x_pt, y_pt)
					# 	try:
					# 		bounding_box = cv2.rectangle(image, x_pt, y_pt, color[i].tolist(), 2)
					# 		img = cv2.add(image, bounding_box)
					# 	except TypeError:
					# 		print('not enough points to make rectangle')
					# 	x_pt = ()
					# 	y_pt = ()
					if i%11==0:
						bounding_box = cv2.rectangle(image, (int(a - box_width/2), int(b - box_height/2)), (int(a + box_width/2), int(b + box_height/2)), color[i].tolist(), 2)
						img = cv2.add(image, bounding_box)
					img = cv2.add(image,mask)

			# recalculate face detection bounding boxes every n seconds (n = 5)
			# if(time.time() - delay_start) >= 5:
			# 	old_frame, boxes = face_detect()
			# 	old_frame_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

			# 	all_points = []
			# 	for box in boxes:
			# 	# 	top_left = [[box[0], box[1]]]
			# 	# 	bottom_right = [[box[2], box[3]]]
			# 	# 	top_right = [[box[0] + (box[2] - box[0]), box[1]]]
			# 	# 	bottom_left = [[box[2] - (box[2] - box[0]), box[3]]]
			# 	# 	all_points.append(top_left)
			# 	# 	all_points.append(bottom_right)
			# 	# 	all_points.append(top_right)
			# 	# 	all_points.append(bottom_left)

			# 	# print(all_points)
			# 	# p0 = np.float32(all_points)

			# 		box_width = (box[2] - box[0])
			# 		box_height = (box[3] - box[1])
			# 		mid_box = [box[0] + box_width / 2, box[1] + box_height / 2]
			# 		all_points.append([mid_box])
			# 	p0 = np.float32(all_points)
			# 	delay_start = time.time()

		    # Now update the previous frame and previous points
			#else:
			old_frame = image.copy()
			old_frame_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
			p0 = good_new.reshape(-1,1,2)

			end_time = time.time()

		
		# Display FPS
		fps = 1 / (end_time - start_time)
		image = cv2.putText(img, 'fps: %.3f' % fps, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))


		cv2.imshow("Face Detect", image)

        # Process and update keycode
		if chr(keyCode) == 'q' or keyCode == 27:
			break
		if chr(keyCode) == 'e':
			encrypt_status = not encrypt_status
		if chr(keyCode) == 'd':
			decrypt_status = not decrypt_status
		if chr(keyCode) == 'v':
			verbose = not verbose

	cap.release()
	cv2.destroyAllWindows()




if __name__ == '__main__':

	if args.jetson:
		cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
	else:
		cap = cv2.VideoCapture(0)

	if cap.isOpened():
		cv2.namedWindow("Face Detect", cv2.WINDOW_AUTOSIZE)

		# load net
		num_classes = 2 # +1 background
		net = build_ssd('test', 300, num_classes) # initialize SSD
		net.load_state_dict(torch.load(args.trained_model, map_location=torch.device('cpu')))
		net.eval()
		print('Finished loading model!')

		if args.cuda:
			net = net.cuda()
			cudnn.benchmark = True

		transformer = BaseTransform(net.size, (104, 117, 123))

        # Variables to control verbosity
		encrypt_status = 1
		decrypt_status = 1
		verbose = 0

		if args.encrypt:
			encryptor = encryption()

		if  args.optical_flow:
			optical_flow()

		if args.multi_tracking:
			multi_tracking()

		else:
			face_detect_every_frame()


	else:
		print("Unable to open camera")




