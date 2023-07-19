import cv2
import face_recognition as fr


# load images
control_picture = fr.load_image_file('PictureA.jpg')
test_picture = fr.load_image_file('PictureC.jpg')

# transform images to RGB
control_picture = cv2.cvtColor(control_picture, cv2.COLOR_BGR2RGB)
test_picture = cv2.cvtColor(test_picture, cv2.COLOR_BGR2RGB)

# locate control face
face_A_location = fr.face_locations(control_picture)[0]
coded_face_A = fr.face_encodings(control_picture)[0]

# locate test face
face_B_location = fr.face_locations(test_picture)[0]
coded_face_B = fr.face_encodings(test_picture)[0]

# frame the control face
cv2.rectangle(control_picture,
              (face_A_location[3], face_A_location[0]),
              (face_A_location[1], face_A_location[2]),
              (0, 255, 0),
              2)

# frame the test face
cv2.rectangle(test_picture,
              (face_B_location[3], face_B_location[0]),
              (face_B_location[1], face_B_location[2]),
              (0, 255, 0),
              2)

# perform comparison
result = fr.compare_faces([coded_face_A], coded_face_B, 0.4)
print(result)

# measurement of distances
distance = fr.face_distance([coded_face_A], coded_face_B)
print(distance)

# display images
cv2.imshow('My Control Picture', control_picture)
cv2.imshow('My Test Picture', test_picture)

# keep the program running
cv2.waitKey(0)

