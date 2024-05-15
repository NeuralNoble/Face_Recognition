import cv2
import face_recognition as fr
font = cv2.FONT_HERSHEY_SIMPLEX

width = 640
height = 360
cap = cv2.VideoCapture(0)

cap.set(3, width)
cap.set(4, height)


aman = fr.load_image_file("Photo on 15-05-24 at 5.46 PM.jpg")
# faceLoc = fr.face_locations(aman)[0]
# faceEnc = fr.face_encodings(aman)[0]

face_locations = fr.face_locations(aman)
if len(face_locations) == 0:
    print("No faces found in the image")
    exit()

# Encode the face
face_encodings = fr.face_encodings(aman, face_locations)
if len(face_encodings) == 0:
    print("Could not encode the face in the image")
    exit()

faceLoc = face_locations[0]
faceEnc = face_encodings[0]

knownEncodings = [faceEnc]
names = ["Aman Anand"]


while True:
    ret, frame = cap.read()
    frameBGR = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    faceLocations = fr.face_locations(frameBGR)
    unkownEncodings = fr.face_encodings(frameBGR, faceLocations)

    for faceLocation,unkownEncoding in zip(faceLocations,unkownEncodings):
        top, right, bottom, left = faceLocation
        cv2.rectangle(frame, (left, top), (right, bottom), (255,0,0),2)
        name = "unknown person"
        matches = fr.compare_faces(knownEncodings,unkownEncoding)
        if True in matches:
            matchIndex = matches.index(True)
            name = names[matchIndex]
        cv2.putText(frame, name, (left,top),font,.75,(0,0,255),2)

    cv2.imshow("Face Recognition", frame)
    cv2.waitKey(1)
