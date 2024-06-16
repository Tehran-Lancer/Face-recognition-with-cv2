import cv2
model_face= "models/haarcascade_frontalface_default.xml"
model_eye= "models/haarcascade_eye.xml"
faceCascade = cv2.CascadeClassifier(model_face)
eyeCascade = cv2.CascadeClassifier(model_eye)
person= cv2.imread("image/body.jpg")
image = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
image_gray = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(
    image_gray,
    scaleFactor=1.1,
    minNeighbors =5,
    minSize=(30, 30))
for (x, y, w, h) in faces:
       cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
eyes = eyeCascade.detectMultiScale(image_gray)
for (xx,yy,ww,hh ) in eyes :
       cv2.rectangle(image,(xx,yy),(xx+ww,yy+hh),(40,120,20),2)
cv2.imshow('YourFace', image)
cv2.waitKey()
cv2.destroyAllWindows()