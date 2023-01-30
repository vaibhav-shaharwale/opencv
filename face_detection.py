import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

img = cv2.imread('margot_will.jpg')

grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(grey_img, scaleFactor = 1.07, minNeighbors = 5)
print(faces)       # co-ordinates of detected faces

for x, y, w, h in faces:
    img = cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 3)

resized_img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
cv2.imshow('windows', resized_img)

k = cv2.waitKey(0)
if k == 1:         # wait for key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('face_detect1.png',img)
    cv2.destroyAllWindows()
    print('image saved')