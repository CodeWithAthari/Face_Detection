import cvzone
import numpy as np
from cv2 import cv2
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.HandTrackingModule import HandDetector
import face_recognition
import pickle
import asyncio
import asyncio

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    detector = FaceDetector(minDetectionCon=0.75)
    print("Loading Encoding File")
    file = open("EncodedImages.p", "rb")
    print("Encoded File Loaded")
    encodedKnownImgListWithIds = pickle.load(file)
    file.close()
    encodedImages, encodedImagesIds = encodedKnownImgListWithIds
    counter = 0
    while True:
        success, img = cap.read()
        img, bboxs = detector.findFaces(img, draw=True)
        # Face Detection with Images
        imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)
        knownFaceLocation = face_recognition.face_locations(imgSmall)
        encodedCurrentFace = face_recognition.face_encodings(imgSmall, knownFaceLocation)
        for encodedFace, faceLocation in zip(encodedCurrentFace, knownFaceLocation):
            matches = face_recognition.compare_faces(encodedImages, encodedFace)
            faceDis = face_recognition.face_distance(encodedImages, encodedFace)
            print("Matches", matches)
            print("Face Dis", faceDis)
            # print("Face Distance", faceDis)
            matchIndex = np.argmin(faceDis)
            # print("MatchIndex ", matchIndex)
            # print("ImageId ",encodedImagesIds[matchIndex])
            if matches[matchIndex]:
                # print("Known Face Detected ", encodedImagesIds[matchIndex])
                imgNew = cv2.imread(f"Images/{encodedImagesIds[matchIndex]}.png")
                imgNew = cv2.resize(imgNew, (100, 100))  # resize imgNew
                print("Face Detected")
                # h, w = imgNew.shape
                # y1, x2, y2, x1 = faceLocation
                # y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                # bbox = y1, x2, y2, x1
                # img = cvzone.cornerRect(img, bbox, rt=0)
                img[0:100, 0:100] = imgNew

            else:
                print("Unknown Face")
                cv2.putText(img, "Unknown Face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
