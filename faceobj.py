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

#remove bg from frame
def remove_background_from_frame(frame, background_image):
    # Ensure that both the frame and background_image have the same dimensions
    if frame.shape[:2] != background_image.shape[:2]:
        raise ValueError("Frame and background_image must have the same dimensions.")

    # Create a mask by thresholding the frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_frame, 1, 255, cv2.THRESH_BINARY)

    # Invert the mask to keep the foreground and remove the background
    mask_inv = cv2.bitwise_not(mask)

    # Extract the foreground from the frame using the mask
    foreground = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # Combine the foreground and background_image to get the final result
    result = cv2.add(foreground, background_image)

    return result

# Check and download YOLOv3 weights
download_yolov3_weights()

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load YOLOv3 weights and configuration file
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Load COCO dataset class names
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Open a video capture stream (0 is usually the default camera)
cap = cv2.VideoCapture(0)
background  = cv2.imread('background1.jpg')
# Define the minimum distance threshold for littering detection (adjust as needed)
min_distance_threshold = 10 # You can change this value

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get the height and width of the frame
    height, width = frame.shape[:2]

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Initialize a list to store objects held in hand
    objects_in_hand = []

    #frame = remove_background_from_frame(frame, background)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Crop the region of interest (ROI) around the face
        roi = frame[y:y + h, x:x + w]
        #show the roi image
        cv2.imshow('roi', roi)
        # Convert the ROI to grayscale for object detection (you can use a more advanced method)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Threshold the ROI to create a binary image (you may need more advanced object detection)
        _, thresholded = cv2.threshold(gray_roi, 128, 255, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Flag objects detected in the hand (you may need to adjust the criteria)
        for contour in contours:
            x_c, y_c, w_c, h_c = cv2.boundingRect(contour)

            x_ext = x - 50
            y_ext = y - 50
            w_ext = w + 100  # 50 units on both sides
            h_ext = h + 100  # 50 units on both sides

            # Check if the object is within the extended bounds of the face ROI
            if x_c > x_ext and y_c > y_ext and x_c + w_c < x_ext + w_ext and y_c + h_c < y_ext + h_ext:
                objects_in_hand.append(contour)

    # Draw rectangles around detected objects (contours) in the hand
    for contour in objects_in_hand:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Object in Hand", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Calculate the distance between the face and the object in hand
    if objects_in_hand:
        object_center_x = x + w // 2
        object_center_y = y + h // 2
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        distance = math.sqrt((object_center_x - face_center_x)**2 + (object_center_y - face_center_y)**2)
        print(distance)
        # Check if the distance crosses the minimum threshold
        if distance > min_distance_threshold:
            cv2.putText(frame, "Littering Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Convert the frame to a blob for YOLOv3 object detection
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Set the input to the YOLOv3 neural network
    net.setInput(blob)

    # Get output layer names
    output_layer_names = net.getUnconnectedOutLayersNames()

    # Forward pass through the YOLOv3 network
    detections = net.forward(output_layer_names)

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Check if the confidence threshold is met (adjust as needed)
            if confidence > 0.5:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)

                # Calculate object coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Check if the object is within the bounds of the face ROI
                if x > 0 and y > 0 and x + w < width and y + h < height:
                    label = f"{classes[class_id]} ({confidence:.2f})"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the frame with both face and object detection
    cv2.imshow("Face and Object Detection", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()