import cv2
import face_recognition as fr
import pickle
font = cv2.FONT_HERSHEY_SIMPLEX

width = 640
height = 360

cam = cv2.VideoCapture(0)

cam.set(3, width)
cam.set(4, height)

with open('train.pkl', 'rb') as f:
    names = pickle.load(f)
    knownEncodings = pickle.load(f)

while True:
    ignore, unknownFace = cam.read()
    unknownFaceBGR = cv2.cvtColor(unknownFace, cv2.COLOR_RGB2BGR)
    faceLocations = fr.face_locations(unknownFaceBGR)
    unknownEncodings = fr.face_encodings(unknownFaceBGR, faceLocations)

    for faceLocation, unknownEncoding in zip(faceLocations, unknownEncodings):
        top, right, bottom, left = faceLocation
        cv2.rectangle(unknownFace, (left, top), (right, bottom), (255, 0, 0), 2)
        name = "unkown name"
        matches = fr.compare_faces(knownEncodings, unknownEncoding)
        if True in matches:
            idx = matches.index(True)
            name = names[idx]
        cv2.putText(unknownFace, name, (left,top),font,0.5,(0,0,255),2)

    cv2.imshow("face",unknownFace)
    cv2.waitKey(1)
