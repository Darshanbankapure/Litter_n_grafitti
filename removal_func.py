import cv2
import numpy as np

def remove_background(video_stream, background_image):
    # Create a VideoCapture object to read the input video stream
    cap = cv2.VideoCapture(video_stream)
    
    # Read the background image
    background = cv2.imread(background_image)
    
    # Define the codec and create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 30.0, (640, 480))  # Adjust resolution as needed
    
    # Create a background subtractor
    # You can use different algorithms like MOG2 or KNN
    # You may need to fine-tune the parameters for your specific use case
    fg_bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply the background subtractor to the current frame
        fg_mask = fg_bg_subtractor.apply(frame)
        
        # Invert the mask to keep the foreground and remove the background
        fg_mask = cv2.bitwise_not(fg_mask)
        
        # Combine the frame and background using the mask
        result = cv2.bitwise_and(frame, frame, mask=fg_mask)
        
        # Resize the background to match the frame size
        background = cv2.resize(background, (frame.shape[1], frame.shape[0]))
        
        # Invert the mask again to keep the background
        bg_mask = cv2.bitwise_not(fg_mask)
        
        # Combine the result and background using the inverted mask
        final_result = cv2.bitwise_and(result, background, mask=bg_mask)
        
        # Write the frame to the output video
        out.write(final_result)
        
        # Display the resulting frame
        cv2.imshow('Output', final_result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release video objects and close windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_stream = 'input_video.mp4'  # Replace with your video stream
    background_image = 'background.jpg'  # Replace with your background image
    remove_background(video_stream, background_image)
