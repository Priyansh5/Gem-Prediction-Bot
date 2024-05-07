import os
import cv2
import numpy as np

def preprocess_image(image_path):
    # Load the image and convert it to float32 for normalization
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Resize the image to a consistent size
    height, width, _ = image.shape
    resize_factor = 224 / min(height, width)
    if height > width:
        new_height = 224
        new_width = int(resize_factor * width)
    else:
        new_width = 224
        new_height = int(resize_factor * height)
    image = cv2.resize(image, (new_width, new_height))

    # Normalize the image pixel values
    image = (image / 255.0) - 0.5

    # Add a batch dimension and return the preprocessed image
    image = np.expand_dims(image, axis=0)
    return image

if __name__ == "__main__":
    # Test the preprocess_image function on a sample image
    image_path = "captured_image.jpg"
    preprocessed_image = preprocess_image(image_path)

    # Print the shape of the preprocessed image
    print("Preprocessed image shape:", preprocessed_image.shape)