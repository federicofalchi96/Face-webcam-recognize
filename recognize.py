import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime

# Cartella con immagini di riferimento
KNOWN_FACES_DIR = "images"
# Cartella per salvare gli screenshot dei riconoscimenti
RECOGNIZED_DIR = "recognized"
os.makedirs(RECOGNIZED_DIR, exist_ok=True)

# Lista dei nomi dei volti noti e dei loro encodings
known_face_encodings = []
known_face_names = []

# Carica immagini di riferimento e genera encodings
for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith((".jpg", ".png")):
        path = os.path.join(KNOWN_FACES_DIR, filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])

# Apri webcam
video_capture = cv2.VideoCapture(0)
print("Premi 'q' per uscire.")

# Set di nomi già riconosciuti in questa sessione
recognized_names_set = set()

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Sconosciuto"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        # Disegna rettangolo e scritta
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 20), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 2, bottom - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Salva screenshot solo se non è già stato salvato
        if name != "Sconosciuto" and name not in recognized_names_set:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{RECOGNIZED_DIR}/{name}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            recognized_names_set.add(name)
            print(f"[INFO] Riconoscimento salvato: {filename}")

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
