import cv2
import time
import urllib.request
import numpy

def detect(frame, classifier, type):

    results = classifier.detectMultiScale3(frame, 1.1, 3, outputRejectLevels=True)
	
    # how many faces have been detected on the frame
    try:
        rects = results[0]
        weights = results[2]
        count = len(rects)
        weight = max(weights)
    except:
        count = 0
        weight = 0
	
    print (str(count) + ' ' + type + ' detected with weight ' + str(weight))

# load pre-trainer classifier
frontalFaceClassifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
profileFaceClassifier = cv2.CascadeClassifier('./haarcascade_profileface.xml')
upperBodyClassifier = cv2.CascadeClassifier('./haarcascade_upperbody.xml')
lowerBodyClassifier = cv2.CascadeClassifier('./haarcascade_lowerbody.xml')
fullBodyClassifier = cv2.CascadeClassifier('./haarcascade_fullbody.xml')

while (True):
	
    # get image from url
    response = urllib.request.urlopen('http://bigbrother/mjpeg_read.php')
    arr = numpy.asarray(bytearray(response.read()), dtype="uint8")
    frame = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

    # pass the frame to the classifier
    detect(frame, frontalFaceClassifier, 'frontal faces')
    detect(frame, profileFaceClassifier, 'profile faces')
    detect(frame, upperBodyClassifier, 'upper bodies')
    detect(frame, lowerBodyClassifier, 'lower bodies')
    detect(frame, fullBodyClassifier, 'full bodies')

    time.sleep(1)
    