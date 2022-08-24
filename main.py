import cv2
import time
import numpy as np
cap = cv2.VideoCapture(0)

# Storing the frame before stating the loop 
# 2 sec delay for adjusting the exposure of the camera
_,background = cap.read()
time.sleep(2)
_,bakcground = cap.read()

# np.ones((5,5),np.uint8) create a 5×5 8 bit integer matrix.
open_kernel = np.ones((5,5),np.uint8)
close_kernel = np.ones((7,7),np.uint8)
dilation_kernel = np.ones((10, 10), np.uint8)

def filter(mask):
    close_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    open_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, close_kernel)
    dilation = cv2.dilate(open_mask, dilation_kernel, iterations = 1)
    return dilation

# check weather the camera is open or not
while cap.isOpened(): 
    _, frame = cap.read()
    # color of the cloak
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([20, 100, 100])
    upper_bound = np.array([30, 255, 255])
    # rest of the background color
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask = filter(mask)
    # finding out the location of the cloak 
    cloak = cv2.bitwise_and(background, background, mask=mask)
    # create inverse mask 
    inverse_mask = cv2.bitwise_not(mask)  

    # Apply the inverse mask to take those region of the current frame where cloak is not present 
    current_background = cv2.bitwise_and(frame, frame, mask=inverse_mask)

    # Combine cloak region and current_background region to get final frame 
    combined = cv2.add(cloak, current_background)

    cv2.imshow("Final output", combined)


    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

