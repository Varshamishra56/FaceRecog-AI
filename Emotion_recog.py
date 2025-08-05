import os
import cv2
from deepface import DeepFace

def detect_emotion_live():
    """
    Detects and displays emotions in real-time using webcam.
    """
    print(" Starting real-time emotion detection... Press 'q' to quit.")
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Failed to capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            try:
                analysis = DeepFace.analyze(face_crop, actions=["emotion"], enforce_detection=False)[0]
                emotion = max(analysis['emotion'], key=analysis['emotion'].get)

                # Draw label and box
                label = f"Emotion: {emotion}"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

            except Exception as e:
                print(" Emotion detection error:", e)

        cv2.imshow("Real-time Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_emotion_live()
