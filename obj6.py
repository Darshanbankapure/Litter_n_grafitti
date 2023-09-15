import cv2
import numpy as np
import math
import os
import urllib.request

# Function to download YOLOv3 weights file if it's not already present
def download_yolov3_weights():
    weights_url = "https://pjreddie.com/media/files/yolov3.weights"
    weights_filename = "yolov3.weights"

    if not os.path.exists(weights_filename):
        print(f"Downloading YOLOv3 weights from {weights_url}")
        urllib.request.urlretrieve(weights_url, weights_filename)
        print("Download complete.")

# Check and download YOLOv3 weights
download_yolov3_weights()

# Load YOLOv3 weights and configuration file
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Load COCO dataset class names
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Open a video capture stream (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Capture a single frame
ret, frame = cap.read()

# Get the height and width of the frame
height, width = frame.shape[:2]

# Convert the frame to a blob for YOLOv3 object detection
blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

# Set the input to the YOLOv3 neural network
net.setInput(blob)

# Get output layer names
output_layer_names = net.getUnconnectedOutLayersNames()

# Forward pass through the YOLOv3 network
detections = net.forward(output_layer_names)

# Iterate through detections and exclude "person" objects
for detection in detections:
    for obj in detection:
        scores = obj[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # Check if the confidence threshold is met (adjust as needed)
        if confidence > 0.8:
            class_name = classes[class_id]

            # Exclude objects labeled as "person"
            if class_name != "person":
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)

                # Calculate object coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                label = f"{class_name} ({confidence:.2f})"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

               # Print the coordinates of the center
                print(f"Object: {classes[class_id]}, Confidence: {confidence:.2f}, Center (x, y): ({center_x}, {center_y})")

    # Display the frame with object detection
    cv2.imshow("Object Detection", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite("detected_objects.jpg", frame)
        break
    # Release the video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
