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
    faces = faceClassifier.detectMultiScale3(frame, 1.1, 3)
    upper_bodies = upperBodyClassifier.detectMultiScale3(frame, 1.1, 3)
    full_bodies = fullBodyClassifier.detectMultiScale3(frame, 1.1, 3)
	
    # how many faces have been detected on the frame
    try:
        face_rects = faces[0]
        face_weights = faces[2]
        face_count = len(face_rects)
        face_weight = max(face_weights)
    except:
        face_count = 0
        face_weight = 0
	
    print (str(face_count) + ' faces detected with weight ' + face_weight)

    time.sleep(1)
    