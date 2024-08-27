import cv2



cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set to a higher resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set to a higher resolution
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
side_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
bodydetection_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
while True:
    ret, frame = cap.read()
    if not ret:
        break
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    side_faces = side_face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    bodydetection = bodydetection_cascade.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5,minSize=(100, 100))

    for (x, y, w, h) in faces:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'front face', (x-50,y-20), font, 1, (0, 0, 255),3, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255, 0), 5)
        cv2.line(frame, (x,y), (x + w, y + h),(0,0,0), 2)
        cv2.line(frame, (x + w, y), (x, y + h), (0,0,0), 2)
        midpoint = [(x+x+w)//2, (y+y+h)//2]
        cv2.circle(frame, (midpoint[0],midpoint[1]), 4, (0,0,255), 3)
        cv2.putText(frame, f"location:{int(midpoint[0]),int(midpoint[1])}", (0,30), font, 1, (0, 0,0 ), 2, cv2.LINE_AA)

    for (fx, fy, fw, fh) in bodydetection:
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 2)
        cv2.putText(frame, 'Full body detected', (10, height - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,cv2.LINE_AA)

    # Display the resulting frame





    if len(faces) == 0:
        for (sx, sy, sw, sh) in side_faces:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, 'side face detected', (10, height - 10), font, 2, (0, 0, 255), 5, cv2.LINE_AA)
            cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 5)





    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
