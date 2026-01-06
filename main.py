from ultralytics import YOLO
import cv2
import time
import os

# Create evidence folder
os.makedirs("evidence", exist_ok=True)

# Load YOLO model
model = YOLO("yolov8n.pt")

# Select camera
# 0 = Laptop webcam
# 1 = External camera (change if needed)
camera_id = int(input("Enter Camera ID (0 = Person Only, 1 = All Objects): "))

cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)

print("\n--- Detection Mode ---")
if camera_id == 0:
    print("PERSON DETECTION MODE")
else:
    print("OBJECT DETECTION MODE")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)
    annotated = frame.copy()

    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]

        # MODE 1: PERSON ONLY
        if camera_id == 0 and label != "person":
            continue

        # Draw box & label
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Save evidence ONLY for person detection
        if camera_id == 0 and label == "person":
            cv2.imwrite(f"evidence/person_{int(time.time())}.jpg", frame)

    cv2.imshow("AI Surveillance System", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
