import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential

# -------------------- 1. Load Model --------------------
with open("facialemotionmodel.json", "r") as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json, custom_objects={"Sequential": Sequential})
model.load_weights("facialemotionmodel.h5")
print("✔ Model Loaded!")

# -------------------- 2. Load Haarcascade --------------------
face_haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# -------------------- 3. Emotion Labels --------------------
emotion_dict = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}

# -------------------- 4. Start Webcam --------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_haar_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.3, minNeighbors=5
    )

    for x, y, w, h in faces:
        roi_gray = gray_frame[y : y + h, x : x + w]
        cropped_img = np.expand_dims(
            np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0
        )
        cropped_img = cropped_img / 255.0

        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        emotion = emotion_dict[maxindex]

        # Draw Rectangle and Emotion Text
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2
        )

    # Show Frame
    cv2.imshow("Real-time Face Emotion Detection", frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
