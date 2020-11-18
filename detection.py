import cv2
import time
import urllib

# load pre-trainer classifier
faceClassifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
upperBodyClassifier = cv2.CascadeClassifier('./haarcascade_fullbody.xml')
fullBodyClassifier = cv2.CascadeClassifier('./haarcascade_upperbody.xml')

while (True):
	
	# get image from url
    response = urllib.urlopen('http://192.168.1.115/mjpeg_read.php')
    image = np.asarray(bytearray(response.read()), dtype="uint8")
    frame = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

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
    