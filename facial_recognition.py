import cv2
import face_recognition as fr


# load images
control_picture = fr.load_image_file('PictureA.jpg')
test_picture = fr.load_image_file('PictureB.jpg')

# transform images to RGB
control_picture = cv2.cvtColor(control_picture, cv2.COLOR_BGR2RGB)
test_picture = cv2.cvtColor(test_picture, cv2.COLOR_BGR2RGB)

# display images
cv2.imshow('My Control Picture', control_picture)
cv2.imshow('My Test Picture', test_picture)

# keep the program running
cv2.waitKey(0)

