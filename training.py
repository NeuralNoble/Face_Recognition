import os
import cv2
import face_recognition as fr
import pickle

imageDir = "demoImages-master/known"
encodings = []
names = []

for root, dirs, files in os.walk(imageDir):
    # print(root)
    # print(dirs)
    # print(files)
    for file in files:
        fullPath = os.path.join(root, file)
        name = os.path.splitext(file)[0]
        names.append(name)
        print(name)
        myPicture = fr.load_image_file(fullPath)
        encoding = fr.face_encodings(myPicture)[0]
        encodings.append(encoding)


with open("train.pkl",'wb') as f:
    pickle.dump(names,f)
    pickle.dump(encodings,f)
