import cv2  # Import the OpenCV library for image processing and computer vision tasks

# Initialize the video capture object to access the default camera (usually webcam)
cap = cv2.VideoCapture(0)

# Set the resolution of the video capture to 1280x720 (HD resolution)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set the frame width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set the frame height

# Load the pre-trained Haar Cascade classifiers for detecting frontal faces, side faces, and full bodies
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
side_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
bodydetection_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Start an infinite loop to continuously capture frames from the camera
while True:
    ret, frame = cap.read()  # Capture each frame from the camera
    if not ret:  # If the frame is not captured properly, exit the loop
        break
    
    # Get the width and height of the captured frame
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Convert the frame to grayscale for better performance in detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect frontal faces, side faces, and full bodies in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    side_faces = side_face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    bodydetection = bodydetection_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))

    # Loop through the detected faces and draw rectangles and other annotations around them
    for (x, y, w, h) in faces:
        font = cv2.FONT_HERSHEY_SIMPLEX  # Choose the font for text annotations
        cv2.putText(frame, 'front face', (x-50, y-20), font, 1, (0, 0, 255), 3, cv2.LINE_AA)  # Label the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)  # Draw a rectangle around the face
        cv2.line(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)  # Draw a diagonal line across the face
        cv2.line(frame, (x + w, y), (x, y + h), (0, 0, 0), 2)  # Draw another diagonal line
        midpoint = [(x + x + w) // 2, (y + y + h) // 2]  # Calculate the midpoint of the face
        cv2.circle(frame, (midpoint[0], midpoint[1]), 4, (0, 0, 255), 3)  # Draw a circle at the midpoint
        cv2.putText(frame, f"location:{int(midpoint[0]), int(midpoint[1])}", (0, 30), font, 1, (0, 0, 0), 2, cv2.LINE_AA)  # Display the location of the midpoint

    # Loop through the detected full bodies and draw rectangles around them
    for (fx, fy, fw, fh) in bodydetection:
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 2)  # Draw a rectangle around the full body
        cv2.putText(frame, 'Full body detected', (10, height - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)  # Label the full body detection

    # If no frontal faces are detected, check for side faces
    if len(faces) == 0:
        for (sx, sy, sw, sh) in side_faces:
            font = cv2.FONT_HERSHEY_SIMPLEX  # Choose the font for text annotations
            cv2.putText(frame, 'side face detected', (10, height - 10), font, 2, (0, 0, 255), 5, cv2.LINE_AA)  # Label the side face
            cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 5)  # Draw a rectangle around the side face

    # Display the processed frame with annotations in a window named 'frame'
    cv2.imshow('frame', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
