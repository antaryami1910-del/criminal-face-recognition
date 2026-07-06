import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import winsound

# ==============================
# STEP 1: LOAD IMAGES
# ==============================
path = 'criminal_images'
images = []
classNames = []

myList = os.listdir(path)
print("Loading images...")

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')

    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

print("Images loaded:", classNames)

# ==============================
# STEP 2: ENCODE FACES
# ==============================
def findEncodings(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)

        if len(encodes) > 0:
            encodeList.append(encodes[0])
        else:
            print("No face found in one image")

    return encodeList

encodeListKnown = findEncodings(images)
print("Encoding Complete")

# ==============================
# STEP 3: CREATE FOLDER
# ==============================
if not os.path.exists("unknown_faces"):
    os.makedirs("unknown_faces")

# ==============================
# STEP 4: START CAMERA
# ==============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera error")
    exit()

# ==============================
# STEP 5: FACE RECOGNITION LOOP
# ==============================
while True:
    success, img = cap.read()

    if not success:
        print("Failed to grab frame")
        break

    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgSmall)
    encodesCurFrame = face_recognition.face_encodings(imgSmall, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):

        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        # Safety check
        if len(faceDis) == 0:
            continue

        matchIndex = np.argmin(faceDis)

        now = datetime.now()
        timeNow = now.strftime('%Y-%m-%d %H:%M:%S')

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            color = (0, 255, 0)

            winsound.Beep(1000, 500)

        else:
            name = "UNKNOWN"
            color = (0, 0, 255)

            filename = f"unknown_faces/unknown_{np.random.randint(1000)}.jpg"
            cv2.imwrite(filename, img)

        # Log file
        with open("log.txt", "a") as f:
            f.write(f"{name} detected at {timeNow}\n")

        # Draw rectangle
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)

        cv2.putText(img, name, (x1 + 6, y2 - 6),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Face Recognition System', img)

    if cv2.waitKey(1) == 13:
        break

# ==============================
# STEP 6: CLEANUP
# ==============================
cap.release()
cv2.destroyAllWindows()