import cv2
import mediapipe as mp
import time

mp_draw = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands
tipIds = [4, 8, 12, 16, 20]

pipeline = mp_hand.Hands(static_image_mode=False,
                          max_num_hands=2,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5)

video = cv2.VideoCapture(0)

while True:
    ret, image = video.read()
    if not ret:
        continue

    image = cv2.flip(image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pipeline.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    lmList = []
    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            myHands = results.multi_hand_landmarks[0]
            for id, lm in enumerate(myHands.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            mp_draw.draw_landmarks(image, hand_landmark, mp_hand.HAND_CONNECTIONS)

    fingers = []
    if len(lmList) != 0:
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        total = fingers.count(1)

        if total == 0:
            cv2.putText(image, "No Finger", (45, 375), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

        elif total == 1:
            cv2.putText(image, "1 Finger", (45, 375), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

        elif total == 2:
            cv2.putText(image, "2 Finger", (45, 375), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

        elif total == 3:
            cv2.putText(image, "3 Finger", (45, 375), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

        elif total == 4:
            cv2.putText(image, "4 Finger", (45, 375), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

        elif total == 5:
            cv2.putText(image, "5 Finger", (45, 375), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    cv2.imshow("Frame", image)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

pipeline.close()
video.release()
cv2.destroyAllWindows()
