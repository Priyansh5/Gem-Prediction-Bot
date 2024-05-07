import os
import cv2
import numpy as np
import random
from PIL import Image, ImageOps

def augment_image(image, label):
    # Apply random transformations to the image
    transformations = [
        # Flip the image horizontally with a 50% probability
        lambda x: cv2.flip(x, 1) if random.random() < 0.5 else x,
        # Randomly rotate the image up to 30 degrees
        lambda x: cv2.warpAffine(
            x,
            cv2.getRotationMatrix2D(
                (x.shape[1] / 2, x.shape[0] / 2),
                random.uniform(-30, 30),
                1,
            ),
            (x.shape[1], x.shape[0]),
        ) if random.random() < 0.5 else x,
        # Randomly shift the image up to 20% of the image size
        lambda x: cv2.pyrMeanShiftFiltering(
            x,
            (int(x.shape[1] * 0.1), int(x.shape[0] * 0.1)),
            (int(x.shape[1] * 0.05), int(x.shape[0] * 0.05)),
            10,
            0.01,
        ) if random.random() < 0.5 else x,
        # Randomly adjust the brightness and contrast of the image
        lambda x: adjust_brightness_contrast(x, random.uniform(0.8, 1.2), random.uniform(0.8, 1.2)),
    ]
    for transformation in transformations:
        image = transformation(image)

    # Convert the label to a one-hot encoded array
    num_classes = 2
    one_hot_label = np.zeros(num_classes)
    one_hot_label[label] = 1

    # Return the augmented image and one-hot encoded label
    return image, one_hot_label

def adjust_brightness_contrast(image, factor, contrast):
    # Apply brightness and contrast adjustments to the image
    image = cv2.addWeighted(
        image,
        factor,
        np.zeros(image.shape, image.dtype),
        0,
        0,
    )
    image = cv2.addWeighted(
        image,
        1,
        np.zeros(image.shape, image.dtype),
        0,
        contrast,
    )
    return image

def save_image(image, label, folder):
    # Save the augmented image to disk in the specified folder
    filename = f"image_{label}.png"
    filepath = os.path.join(folder, filename)
    cv2.imwrite(filepath, image)

if __name__ == "__main__":
    # Test the augment_image function on a sample image
    image_path = "preprocessed_image.npy"
    label = 1
    image = np.load(image_path)

    # Augment the image and print the augmented image shape
    augmented_image, one_hot_label = augment_image(image, label)
    print("Augmented image shape:", augmented_image.shape)

    # Save the augmented image to disk
    folder = "augmented_images"
    save_image(augmented_image, label, folder)