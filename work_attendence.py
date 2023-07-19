import cv2
import face_recognition as fr
import os


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


# encode images
def encode(images):

    # create new list
    encoded_list = []

    # convert images to RBG
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # encode
        encoded_img = fr.face_encodings(image)[0]

        # add to the list
        encoded_list.append(encoded_img)

    # return encoded list
    return encoded_list


encoded_member_list = encode(my_images)


# take webcam picture
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# read captured image
