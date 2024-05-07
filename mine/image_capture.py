import os
import mss
import cv2
import time

def capture_image():
    # Set up the monitor region for screenshot capture
    monitor = {"top": 100, "left": 100, "width": 800, "height": 600}

    # Use mss to capture the screenshot
    with mss.mss() as sct:
        screenshot = sct.grab(monitor)

    # Convert the screenshot to a numpy array and save it as a jpeg image
    image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGBA2BGR)
    file_name = "captured_image_{}.jpg".format(time.strftime("%Y%m%d_%H%M%S"))
    cv2.imwrite(file_name, image)

    # Print a success message to the console
    print(f"Image captured and saved as {file_name}")

if __name__ == "__main__":
    while True:
        capture_image()
        time.sleep(1) # Wait for 1 second between captures