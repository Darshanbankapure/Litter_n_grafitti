import cv2
import numpy as np

# Load YOLOv3 weights and configuration
yolo_net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')  # Replace with your YOLOv3 weights and configuration files
yolo_classes = []
with open('coco.names', 'r') as file:  # Replace with your class names file (e.g., coco.names)
    yolo_classes = file.read().strip().split('\n')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize variables to track object and hand positions
object_detected = False
object_position = None
hand_position = None
object_stopped_frames = 0
object_stopped_threshold = 50  # Adjust as needed
face_height_threshold = 100  # Adjust as needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Create a blob from the frame and perform forward pass
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    detections = yolo_net.forward()

    for detection in detections:
        for obj in detection:
            x, y, width, height, confidence, *class_probs = obj

        if confidence > 0.5:  # Adjust the confidence threshold as needed
            class_id = np.argmax(class_probs)
            class_name = yolo_classes[class_id]

            if class_name == 'hand':
                # Extract hand position from the detected bounding box
                center_x = int(x * width)
                center_y = int(y * height)
                hand_position = (center_x, center_y)

            elif class_name == 'object':
                # Extract object position from the detected bounding box
                center_x = int(x * width)
                center_y = int(y * height)
                object_position = (center_x, center_y)

                # Check the separation between the object and hand
                if hand_position:
                    separation = np.linalg.norm(np.array(hand_position) - np.array(object_position))
                    if separation > face_height_threshold and object_stopped_frames > object_stopped_threshold:
                        print("Littering detected!")
                            # You can perform further actions here, such as saving images or sending alerts

    # Check if the object is stationary
    if hand_position and object_position:
        if np.linalg.norm(np.array(hand_position) - np.array(object_position)) < 10:
            object_stopped_frames += 1
        else:
            object_stopped_frames = 0

    cv2.imshow('Webcam Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
