import os
from typing import Union, Iterator

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from sklearn.svm import LinearSVC

def generate_hog_features(folder_name: str) -> Union[list, list]:
    print("Generating training set feature vectors ...")

    img_path = f"input/{folder_name}"

    x_train = []
    y_train = []

    # Get all the image folder paths
    img_paths = os.listdir(img_path)
    for path in img_paths:

        # Get lists containing names of all images for training set
        main_path = f"{img_path}/{path}"
        all_images = os.listdir(main_path)

        # Run HoGs on training set
        for image in all_images:

            # Get image
            image_path = f"{main_path}/{image}"

            # Read image and resize to 64 x 128 - HoGs requires 1:2 ratio
            image = cv2.imread(image_path)
            image = cv2.resize(image, (64, 128))

            # Calculate HOG descriptor for each image
            hog_desc = feature.hog(
                image,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                transform_sqrt=True,
                block_norm="L2",
            )

            # Add images and labels
            x_train.append(hog_desc)
            y_train.append(path)

    return x_train, y_train


def generate_svm(x_train: list, y_train: list) -> LinearSVC:
    print("Training SVM model ...")
    svm_model = LinearSVC(random_state=42, tol=1e-5)
    svm_model.fit(x_train, y_train)
    return svm_model


def sliding_window(image: np.ndarray, step_sz: int, window_sz: tuple) -> Iterator[tuple]:
    for y in range(0, image.shape[0], step_sz):
        for x in range(0, image.shape[1], step_sz):
            yield (x, y, image[y : y + window_sz[1], x : x + window_sz[0]])


def pyramid(image: np.ndarray, scale=2.5, minSize=(5, 5)) -> Iterator[np.ndarray]:
    yield image

    while True:
        # Calculate new dimensions and resize image
        width = int(image.shape[1] / scale)
        height = int(image.shape[0] / scale)
        dim = (width, height)

        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        # Break when smaller than minimum size
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        yield image


def test_set_algo(image: np.ndarray, svm_model: LinearSVC, step_sz=8, window_sz=(100, 100)) -> Union[str, np.ndarray, np.ndarray]:
    x_test = []
    y_test = []
    image_meta = []
    image = cv2.resize(image, (200, 200))
    win_width, win_height = window_sz

    for (i, image_pyr) in enumerate(pyramid(image)):
        for x, y, window in sliding_window(
            image_pyr, step_sz=step_sz, window_sz=window_sz
        ):
            # Ignore if window does not match shape
            if window.shape[0] != win_height or window.shape[1] != win_width:
                continue

            # Resize to HoG size - 1:2 ratio
            window = cv2.resize(window, (64, 128))
            (hog_desc, hog_image) = feature.hog(
                window,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm="L2",
                visualize=True,
                transform_sqrt=True,
            )

            x_pred = hog_desc.reshape(1, -1)
            y_pred = svm_model.predict(x_pred)[0]

            x_test.append(x_pred)
            y_test.append(y_pred)
            image_meta.append((x, y, hog_image))

    # Get idx of best fitting point
    des_func = list(map(svm_model.decision_function, x_test))
    des_max = max(des_func)

    print("Best Decision: ", str(des_max))

    des_func_idx = des_func.index(des_max)
    label = y_test[des_func_idx]

    # Get best fitting sliding window data
    x, y, hog_image = image_meta[des_func_idx]

    # Draw bounding box on image
    image = cv2.rectangle(
        image, (x, y), (x + win_width, y + win_height), (0, 255, 0), 2
    )

    return label, image, hog_image


def run_svm(folder_name: str, svm_model: LinearSVC) -> None:
    # Run SVM model on test images
    print("Running SVM model on testing set ...")

    img_path = folder_name
    img_path = f"test_images/{img_path}"
    img_paths = os.listdir(img_path)

    output_path = f"outputs/"

    # loop over the test dataset folders
    for (i, img) in enumerate(img_paths):
        # File pathing
        img = f"{img_path}/{img}"
        image = cv2.imread(img)

        # Run HoGs algorithim, sliding window and pyramid
        label, image, hog_image = test_set_algo(image, svm_model)

        # Print predictions
        img_name = img.split("/")[-1][:-4]

        print("Actual:", img_name)
        print("Predicted:", label)

        # Rescale HoG image
        hog_image = hog_image.astype("float64")

        # Add text to image
        actual_text = f"Actual: {img_name}"
        predicted_text = f"Predicted: {label}"

        cv2.putText(
            image,
            predicted_text,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )
        cv2.putText(
            image, actual_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
        )

        # Write output images to output folder
        cv2.imwrite(f"{output_path}hog_{i}.jpg", hog_image * 255.0)
        cv2.imwrite(f"{output_path}pred_{i}.jpg", image)

    return


if __name__ == "__main__":
    # Point to folder
    folder_name = "shapes"

    # Get x and y test feature data
    x_train, y_train = generate_hog_features(folder_name)

    # Create SVM and train data
    svm_model = generate_svm(x_train, y_train)

    # Run testing set
    run_svm(folder_name, svm_model)
