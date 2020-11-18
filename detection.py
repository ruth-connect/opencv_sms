import cv2
import time

# load pre-trainer classifier
classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# read frame-by-frame
frame = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)

# pass the frame to the classifier
persons_detected = classifier.detectMultiScale(frame, 1.3, 5)
	
# how many people have been detected on the frame
try:
	human_count = persons_detected.shape[0]
except:
	human_count = 0

print (str(human_count) + ' humans detected')