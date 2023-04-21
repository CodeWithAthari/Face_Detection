import cv2
import face_recognition
import pickle
import os


def findEncodings(imgList):
    encodeList = []
    print("Encoding Started")
    for i, img in enumerate(imgList) :
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodedImg = face_recognition.face_encodings(img)[0]
        encodeList.append(encodedImg)
        print("Saved ",i)
    print("Encoding Finished")
    return encodeList


if __name__ == "__main__":
    folderPath = "Images"
    pathList = os.listdir(folderPath)
    print(pathList)
    imgList = []
    imgIds = []
    for path in pathList:
        imgList.append(cv2.imread(os.path.join(folderPath, path)))
        imgIds.append(os.path.splitext(path)[0])
    # print(len(imgList))
    # print(imgIds)

    encodedImgList = findEncodings(imgList)
    encodedImgListWithIds = [encodedImgList, imgIds]
    file = open("EncodedImages.p", "wb")
    pickle.dump(encodedImgListWithIds, file)
    file.close()
    print("File Saved")