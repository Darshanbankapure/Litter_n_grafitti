import cv2
import numpy as np

# Load the pre-trained MobileNet-SSD model
prototxt_path = 'deploy.prototxt'
caffemodel_path = 'mobilenet_iter_73000.caffemodel'

net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# Open a video capture stream (0 is usually the default camera)
cap = cv2.VideoCapture(0)

frame_count = 0  # Initialize frame counter

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Process and draw bounding boxes on detected objects
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.8:  # Filter out low-confidence detections
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Save the processed frame as an image (you can adjust the path and format)
    frame_count += 1
    #image_filename = f"frame/frame_{frame_count}.jpg"
    #cv2.imwrite(image_filename, frame)

    # Display the frame with bounding boxes
    cv2.imshow("Object Detection", frame)

    # Break the loop on pressing the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
