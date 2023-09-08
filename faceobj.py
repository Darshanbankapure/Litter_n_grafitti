import cv2
import numpy as np
import math

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load YOLOv3 weights and configuration file
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Load COCO dataset class names
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Open a video capture stream (0 is usually the default camera)
cap = cv2.VideoCapture(0)

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

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Crop the region of interest (ROI) around the face
        roi = frame[y:y + h, x:x + w]

        # Convert the ROI to grayscale for object detection (you can use a more advanced method)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Threshold the ROI to create a binary image (you may need more advanced object detection)
        _, thresholded = cv2.threshold(gray_roi, 128, 255, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Flag objects detected in the hand (you may need to adjust the criteria)
        for contour in contours:
            x_c, y_c, w_c, h_c = cv2.boundingRect(contour)

            # Check if the object is within the bounds of the face ROI
            if x_c > x and y_c > y and x_c + w_c < x + w and y_c + h_c < y + h:
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
