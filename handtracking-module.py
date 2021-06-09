import cv2
import mediapipe as mp

import time


class HandDetector:

    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5,
                 track_confidence=0.5):
        self.mode = mode  # we are creating an object, the obj has its own variable (mode)
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands,
                                         self.detection_confidence, self.track_confidence)
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, landmarks, self.mp_hands.HAND_CONNECTIONS)

        return img

            # for id, lm in enumerate(landmarks.landmark):
            #     print(id, lm)
            #     height, width, c = img.shape
            #     cx, cy = int(lm.x * width), int(lm.y * height)
            #
            #     if id == 0:
            #         cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)



def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    prev_time = 0
    current_time = 0

    while True:
        success, img = cap.read()
        img = detector.find_hands(img)

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        cv2.putText(img, str(int(fps)), (10, 70,), cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 0, 255), 3)

        cv2.imshow('Image', img)
        cv2.waitKey(1)



if __name__ == "__main__":
    main()






