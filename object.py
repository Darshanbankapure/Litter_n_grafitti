import cv2
import numpy as np
import math

# Load SSD model and configuration
net = cv2.dnn.readNet('ssd_mobilenet_v2_coco.pb', 'ssd_mobilenet_v2_coco.pbtxt')

# Load COCO class names (for object labels)
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Initialize the video capture
cap = cv2.VideoCapture(0)  # 0 for the default camera, you can specify a video file path too

# Define littering threshold distance (adjust as needed)
littering_threshold = 100

# Dictionary to store object assignments to people
object_assignments = {}

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    height, width = frame.shape[:2]

    # Prepare the frame for SSD object detection
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    # Set the blob as input to the network
    net.setInput(blob)

    # Run SSD object detection
    detections = net.forward()

    for detection in detections:
        for obj in detection[0, 0, :, :]:
            confidence = obj[2]

            if confidence > 0.5:
                class_id = int(obj[1])
                label = classes[class_id]

                if label == "person":
                    x, y, w, h = (obj[3:7] * np.array([width, height, width, height])).astype(int)

                    # Calculate distance to the camera (assuming camera at the center bottom)
                    distance = calculate_distance(x + w // 2, y + h, width // 2, height)

                    # Check for littering
                    person_id = f"{x}-{y}"
                    if person_id in object_assignments:
                        prev_x, prev_y, _ = object_assignments[person_id]
                        prev_distance = calculate_distance(prev_x, prev_y, width // 2, height)
                        if distance - prev_distance > littering_threshold:
                            cv2.putText(frame, "Littering Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    object_assignments[person_id] = (x, y, label)

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with objects and littering information
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
