import cv2
import face_recognition as fr
import os
import numpy
from datetime import datetime


# create database
path = 'blackpink_members'
my_images = []
members_names = []
members_list = os.listdir(path)

# build list of image paths and image names
for name in members_list:
    this_image = cv2.imread(f'{path}\\{name}')
    my_images.append(this_image)
    members_names.append(os.path.splitext(name)[0])

print(members_list)


# encode images
def encode(images):
    """
    encode member images from folder
    """

    # create new list
    encoded_list = []

    # convert images to RGB
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # encode
        encoded_img = fr.face_encodings(image)[0]

        # add to the list
        encoded_list.append(encoded_img)

    # return encoded list
    return encoded_list


# record attendance
def record_attendance(person):
    """
    read the existing file and add existing names to list
    if captured image contains person not in list, add their name and time to the list
    """
    f = open('register.csv', 'r+')
    data_list = f.readline()
    register_names = []

    for line in data_list:
        newcomer = line.split(',')
        register_names.append(newcomer[0])

    if person not in register_names:
        right_now = datetime.now()
        string_right_now = right_now.strftime('%H:%M:%S')
        f.writelines(f'\n{person},{string_right_now}')


# ecode images from folder containing images of members
encoded_member_list = encode(my_images)

# take webcam picture (captured image)
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# read captured image
success, image = capture.read()

if not success:
    print("Capture image could not be taken")
else:
    # locate the face in captured image
    captured_face = fr.face_locations(image)

    # encode captured face
    encoded_captured_face = fr.face_encodings(image, captured_face)

    # search for match
    for face, location_face in zip(encoded_captured_face, captured_face):
        matches = fr.compare_faces(encoded_member_list, face)
        distances = fr.face_distance(encoded_member_list, face)

        print(distances)

        # get the index of member based on lowest distance
        match_index = numpy.argmin(distances)

        # show coincidences if any
        # adjustable distance (tolerance), default is 0.6, lower the stricter
        if distances[match_index] > 0.6:
            print("Does not match any of our employees")
        else:
            # search for name of matching member
            member_name = members_names[match_index]

            y1, x2, y2, x1 = location_face

            # create rectangle around face
            cv2.rectangle(image,
                          (x1, y1),
                          (x2, y2),
                          (0, 255, 0),
                          2)

            # create name box
            cv2.rectangle(image,
                          (x1, y2 - 35),
                          (x2, y2),
                          (0, 255, 0),
                          cv2.FILLED)

            # add name to box
            cv2.putText(image,
                        member_name,
                        (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (255, 255, 255),
                        2)

            # call the record attendance function
            record_attendance(member_name)

            # show the image captured from webcam
            cv2.imshow('Web Image', image)

            # keep program running
            cv2.waitKey(0)
