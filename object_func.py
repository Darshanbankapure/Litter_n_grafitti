import cv2
import numpy as np

def detect_objects(image_path):
    # Load YOLOv3 weights and configuration file
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

    # Load COCO dataset class names
    with open("coco.names", "r") as f:
        classes = f.read().strip().split("\n")
        
    # Read the input image
    image = cv2.imread(image_path)

    # Get the height and width of the image
    height, width = image.shape[:2]

    # Create a blob from the image for YOLOv3 input
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Set the input to the YOLOv3 neural network
    net.setInput(blob)

    # Get output layer names
    output_layer_names = net.getUnconnectedOutLayersNames()

    # Initialize a list to store the coordinates (x, y) of detected object centers
    object_centers = []

    # Forward pass through the YOLOv3 network
    detections = net.forward(output_layer_names)

    # Process the detections and collect the coordinates
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

                # Append the object center coordinates to the list
                object_centers.append((center_x, center_y))

    # Return the list of object centers
    return object_centers

ans = detect_objects('hk.jpeg')