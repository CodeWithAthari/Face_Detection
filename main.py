from cv2 import cv2
from cvzone.FaceDetectionModule import FaceDetector

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(16, 1280)
    cap.set(9, 720)
    detector = FaceDetector(minDetectionCon=0.75)

    while True:
        success, img = cap.read()
        # img = cv2.flip(img, 1)
        img, bboxs = detector.findFaces(img, draw=True)
        if bboxs:
            for i, bbox in enumerate(bboxs):
                x, y, w, h = bbox["bbox"]
                if x < 0: x = 0
                if y < 0: y = 0
                imgCrop = img[y:y + h, x:x + w]
                print(img[y:y + h, x:x + w])
                imgBlur = cv2.blur(imgCrop, (100, 100))
                img[y:y + h, x:x + w] = imgBlur
                # cv2.imshow(f"Image Cropoed {i}", imgCrop)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
