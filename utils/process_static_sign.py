import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

detector = HandDetector(maxHands=2)
classifier = Classifier("keras_model.h5", "labels.txt") 
labels = ['call', 'friend', 'house', 'i_love_you', 'love', 'more', 'no', 'what', 'you']

offset = 20
imgWidth = 600
imgHeight = 400

def process_static_sign(img):
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    index = None

    imgWhite = np.ones((imgHeight, imgWidth, 3), np.uint8) * 255

    if hands:
        aspectRatio = None
        if len(hands) >= 2:
            hand_boxes = [hand['bbox'] for hand in hands]
            x_min = min(box[0] for box in hand_boxes)
            y_min = min(box[1] for box in hand_boxes)
            x_max = max(box[0] + box[2] for box in hand_boxes)
            y_max = max(box[1] + box[3] for box in hand_boxes)

            imgCrop = img[y_min - offset:y_max + offset, x_min - offset:x_max + offset]
            imgCropShape = imgCrop.shape

            aspectRatio = (y_max - y_min) / (x_max - x_min)

        elif len(hands) == 1:
            x_min, y_min, w, h = hands[0]['bbox']
            x_max, y_max = x_min + w, y_min + h
            imgCrop = img[y_min - offset:y_max + offset, x_min - offset:x_max + offset]
            imgCropShape = imgCrop.shape

            aspectRatio = h / w

        if not imgCrop.size == 0:
            if aspectRatio > 1:
                h_start = (imgHeight - imgCropShape[0]) // 2
                h_end = h_start + imgCropShape[0]
                w_start = (imgWidth - imgCropShape[1]) // 2
                w_end = w_start + imgCropShape[1]

                if h_start >= 0 and h_end <= imgHeight and w_start >= 0 and w_end <= imgWidth:
                    imgWhite[h_start:h_end, w_start:w_end] = imgCrop
            else:
                h_start = (imgHeight - imgCropShape[0]) // 2
                h_end = h_start + imgCropShape[0]
                w_start = (imgWidth - imgCropShape[1]) // 2
                w_end = w_start + imgCropShape[1]

                if h_start >= 0 and h_end <= imgHeight and w_start >= 0 and w_end <= imgWidth:
                    imgWhite[h_start:h_end, w_start:w_end] = imgCrop

            prediction, index = classifier.getPrediction(imgWhite, draw=False)

            if prediction[index] > 0.80:
                cv2.putText(imgOutput, labels[index], (x_min, y_min -20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
                print(prediction, index)

    return imgOutput