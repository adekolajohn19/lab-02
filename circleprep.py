import cv2
import numpy as np

# Get the webcam feed
cap = cv2.VideoCapture(0)

# Create a blue color filter
hsv_min = np.array([90, 100, 100])
hsv_max = np.array([130, 255, 255])

# While the webcam is running
while True:

    # Get a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Apply the blue color filter
    inRange = cv2.inRange(hsv, hsv_min, hsv_max)

    # Find all the blobs in the frame
    contours, hierarchy = cv2.findContours(inRange, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the blobs by size, largest to smallest
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Get the biggest 3 blobs
    biggest_blobs = contours[:3]

    # Draw the biggest 3 blobs on the frame
    for contour in biggest_blobs:
        x, y, w, h = cv2.boundingRect(contour)
        imageFrame = cv2.rectangle(imageFrame, (x, y),
									(x + w, y + h),
									(255, 0, 0), 2)
			
        cv2.putText(imageFrame, "Blue", (x, y),
						cv2.FONT_HERSHEY_SIMPLEX,
						1.0, (255, 0, 0))

    # Display the frame
    cv2.imshow("Frame", frame)

    # If the user presses the Esc key, stop the loop
    if cv2.waitKey(1) == 27:
        break

# Release the webcam
cap.release()

# Close all windows
cv2.destroyAllWindows()
