import os
import cv2
import numpy as np
from deepface import DeepFace

# Set main dataset directory
DATASET_DIR = "Dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

def create_dataset(name, samples=50):
    """
    Captures images from webcam and saves face crops to disk.
    """
    person_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    count = 0

    print(f"Capturing faces for {name}. Press 'q' to quit early.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Cannot access webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = frame[y:y+h, x:x+w]
            save_path = os.path.join(person_dir, f"{name}_{count}.jpg")
            cv2.imwrite(save_path, face_img)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow("Capturing Faces", frame)

            if count >= samples:
                print(f" Collected {samples} face samples for {name}")
                cap.release()
                cv2.destroyAllWindows()
                return

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def create_embeddings():
    """
    Extracts FaceNet embeddings for all face images in dataset.
    """
    print(" Creating embeddings...")
    embeddings = {}

    for person in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(person_dir):
            continue

        embeddings[person] = []
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            try:
                rep = DeepFace.represent(img_path=img_path, model_name="Facenet", enforce_detection=False)
                embeddings[person].append(rep[0]["embedding"])
            except Exception as e:
                print(f"❌ Skipped {img_name}: {e}")

    np.save("embeddings.npy", embeddings)
    print(" Embeddings saved to embeddings.npy")
    return embeddings

def recognize_faces(embeddings):
    """
    Starts webcam and identifies faces using DeepFace + cosine similarity.
    """
    print(" Starting real-time face recognition...")
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            try:
                # Analyze age, gender, emotion
                analysis = DeepFace.analyze(face_crop, actions=["age", "gender", "emotion"], enforce_detection=False)[0]
                age = int(analysis['age'])
                gender = analysis['gender'] if isinstance(analysis['gender'], str) else max(analysis['gender'], key=analysis['gender'].get)
                emotion = max(analysis['emotion'], key=analysis['emotion'].get)

                # Represent face
                face_vec = DeepFace.represent(face_crop, model_name='Facenet', enforce_detection=False)[0]['embedding']

                best_match = "Unknown"
                best_score = -1

                # Compare with known embeddings
                for name, embs in embeddings.items():
                    for emb in embs:
                        similarity = np.dot(face_vec, emb) / (np.linalg.norm(face_vec) * np.linalg.norm(emb))
                        if similarity > best_score:
                            best_score = similarity
                            best_match = name

                label = f"{best_match} ({best_score:.2f})" if best_score > 0.7 else "Unknown"
                info = f"{label}, Age: {age}, Gender: {gender}, Emotion: {emotion}"

                cv2.putText(frame, info, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)

            except Exception as e:
                print(" Recognition error:", e)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print(" Face Recognition Menu")
    print("1. Create Face Dataset")
    print("2. Train & Save Embeddings")
    print("3. Recognize Faces (Real-time)")
    choice = input("Enter your choice (1/2/3): ")

    if choice == '1':
        person_name = input("Enter person's name: ")
        create_dataset(person_name)

    elif choice == '2':
        create_embeddings()

    elif choice == '3':
        if os.path.exists("embeddings.npy"):
            embeddings = np.load("embeddings.npy", allow_pickle=True).item()
            recognize_faces(embeddings)
        else:
            print(" Please train the dataset first using option 2.")

    else:
        print(" Invalid choice. Please enter 1, 2, or 3.")
