import cv2
import numpy as np

# Load YOLOv3 weights and configuration file
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Load COCO dataset class names
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Load the input image
image_path = 'hk.jpeg'  # Replace with the path to your input image
image = cv2.imread(image_path)

# Get the height and width of the image
height, width = image.shape[:2]

# Create a blob from the image for YOLOv3 input
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

# Set the input to the YOLOv3 neural network
net.setInput(blob)

# Get output layer names
output_layer_names = net.getUnconnectedOutLayersNames()

# Forward pass through the YOLOv3 network
detections = net.forward(output_layer_names)

# Process and display the detections
for detection in detections:
    for obj in detection:
        scores = obj[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # Check if the confidence threshold is met (adjust as needed)
        if confidence > 0.4:
            center_x = int(obj[0] * width)
            center_y = int(obj[1] * height)
            w = int(obj[2] * width)
            h = int(obj[3] * height)

            # Calculate object coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            label = f"{classes[class_id]} ({confidence:.2f})"
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Display the image with object detection
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
