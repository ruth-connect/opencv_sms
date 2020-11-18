import cv2
import time
import urllib.request
import numpy

# load pre-trainer classifier
faceClassifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
upperBodyClassifier = cv2.CascadeClassifier('./haarcascade_fullbody.xml')
fullBodyClassifier = cv2.CascadeClassifier('./haarcascade_upperbody.xml')

while (True):
	
	# get image from url
    response = urllib.request.urlopen('http://192.168.1.115/mjpeg_read.php')
    image = numpy.asarray(bytearray(response.read()), dtype="uint8")
    frame = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

    # pass the frame to the classifier
    faces_detected, faces_weights = faceClassifier.detectMultiScale(frame, 1.1, 3)
    upper_bodies_detected, upper_bodies_weights = upperBodyClassifier.detectMultiScale(frame, 1.1, 3)
    full_bodies_detected, full_bodies_weights = fullBodyClassifier.detectMultiScale(frame, 1.1, 3)
	
    # how many faces have been detected on the frame
    try:
        face_count = faces_detected.shape[0]
        face_weight = faces_weights[0]
    except:
        face_count = 0
        face_weight = 0
	
# how many upper bodies have been detected on the frame
    try:
        upper_body_count = upper_bodies_detected.shape[0]
        upper_body_weight = upper_body_weights[0]
    except:
        upper_body_count = 0
        upper_body_weight = 0
        
    # how many full bodies have been detected on the frame
    try:
        full_body_count = full_bodies_detected.shape[0]
        full_body_weight = full_body_weights[0]
    except:
        full_body_count = 0
        full_body_weight = 0

    print (str(face_count) + ' faces detected with weight ' + face_weight)
    print (str(upper_body_count) + ' upper bodies detected with weight ' + upper_body_weight)
    print (str(full_body_count) + ' full bodies detected with weight ' + full_body_weight)

    time.sleep(1)
    