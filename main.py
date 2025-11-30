import cv2

cap = cv2.VideoCapture('youtube.mp4')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
fullbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

while(True):
    #doc lan luot cac frame tu camera
    ret, frame = cap.read()

    #frame = cv2.flip(frame, 1)

    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    bodies = fullbody_cascade.detectMultiScale(gray,
                                  scaleFactor=1.1,
                                  minNeighbors=5,
                                  minSize=(30,30))

    for body in bodies:
        x, y, w, h = body
        body_crop = gray[y:y+h, x:x+w] #do tren numpy nen y truoc x sau

        

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
        cv2.putText(frame, 'Person', (x+w+3, y+h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        faces = face_cascade.detectMultiScale(body_crop)
        for face in faces:
            fx, fy, fw, fh = face
            cv2.rectangle(frame, (fx+x, fy+y), (fx+fw+x, fy+fh+y), (0, 255, 0), 3)
            cv2.putText(frame, 'face', (fx+fw+x+3, fy+fh//2+y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)


    cv2.imshow('camera streaming', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()