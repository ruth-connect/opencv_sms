import cv2
import time

# load pre-trainer classifier
faceClassifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
upperBodyClassifier = cv2.CascadeClassifier('./haarcascade_fullbody.xml')
fullBodyClassifier = cv2.CascadeClassifier('./haarcascade_upperbody.xml')

while (True):
	
    # read frame-by-frame
    frame = cv2.imread('https://64.media.tumblr.com/18bcea114fb85d4abced1b42c53fe37c/tumblr_o0cgt8atDu1v2ia4ro1_1280.jpg', cv2.IMREAD_GRAYSCALE)

    # pass the frame to the classifier
    faces_detected = faceClassifier.detectMultiScale(frame, 1.1, 3)
    upper_bodies_detected = upperBodyClassifier.detectMultiScale(frame, 1.1, 3)
    full_bodies_detected = fullBodyClassifier.detectMultiScale(frame, 1.1, 3)
	
    # how many faces have been detected on the frame
    try:
	    face_count = faces_detected.shape[0]
    except:
	    face_count = 0
	
# how many upper bodies have been detected on the frame
    try:
	    upper_body_count = upper_bodies_detected.shape[0]
    except:
	    upper_body_count = 0

    # how many full bodies have been detected on the frame
    try:
    	full_body_count = full_bodies_detected.shape[0]
    except:
    	full_body_count = 0

    print (str(face_count) + ' faces detected')
    print (str(upper_body_count) + ' upper bodies detected')
    print (str(full_body_count) + ' full bodies detected')

    time.sleep(1)
    