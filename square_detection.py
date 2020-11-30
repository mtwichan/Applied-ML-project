import os
from typing import Union, Iterator

import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from skimage import feature
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Create HoG Features
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

            # Read image
            image = cv2.imread(image_path)

            # Apply Gaussian Blur
            image = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)

            # Resize to 64 x 128 - HoGs requires 1:2 ratio
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

    print("Finished generating training set feature vectors ...")
    return x_train, y_train


# Create the SVM model
def generate_svm(x_train: list, y_train: list) -> LinearSVC:
    print("Training SVM model ...")

    svm_model = LinearSVC(random_state=42, tol=1e-5)
    svm_model.fit(x_train, y_train)

    print("Finished training SVM model ...")
    return svm_model


# Sliding window for test set
def sliding_window(
    image: np.ndarray, step_sz: int, window_sz: tuple
) -> Iterator[tuple]:
    for y in range(0, image.shape[0], step_sz):
        for x in range(0, image.shape[1], step_sz):
            yield (x, y, image[y : y + window_sz[1], x : x + window_sz[0]])


# Perform image pyramid on test set
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


# Algo for test set with sliding window
def test_set_algo(
    image: np.ndarray, svm_model: LinearSVC, step_sz=8, window_sz=(100, 100)
) -> Union[str, str, np.ndarray, np.ndarray]:
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

            # Apply Gaussian Blur
            window = cv2.GaussianBlur(window, (5, 5), cv2.BORDER_DEFAULT)

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

    x_test = x_test[des_func_idx]
    y_test = y_test[des_func_idx]

    # Get best fitting sliding window data
    x, y, hog_image = image_meta[des_func_idx]

    # Draw bounding box on image
    image = cv2.rectangle(
        image, (x, y), (x + win_width, y + win_height), (0, 255, 0), 2
    )

    return x_test, y_test, image, hog_image


# Algo for test set without sliding window
def test_set_algo_raw(
    image: np.ndarray, svm_model: LinearSVC
) -> Union[str, str, np.ndarray, np.ndarray]:

    image = cv2.resize(image, (200, 200))

    # Resize to HoG size - 1:2 ratio
    image = cv2.resize(image, (64, 128))

    # Apply Gaussian Blur
    image = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)

    (hog_desc, hog_image) = feature.hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2",
        visualize=True,
        transform_sqrt=True,
    )

    x_test = hog_desc.reshape(1, -1)
    y_test = svm_model.predict(x_test)[0]

    return x_test, y_test, image, hog_image


# Run the SVM model
def run_svm(folder_name: str, svm_model: LinearSVC, window_pyr=True) -> None:
    # Run SVM model on test images
    print("Running SVM model on testing set ...")

    x_test_all = []
    y_test_all = []

    img_path = f"test_images/{folder_name}"
    img_paths = os.listdir(img_path)

    output_path = f"outputs/"

    # loop over the test dataset folders
    for (i, img) in enumerate(img_paths):
        # File pathing
        img = f"{img_path}/{img}"
        image = cv2.imread(img)

        # Run HoGs algorithim, sliding window and pyramid
        if window_pyr:
            x_test, y_test, image, hog_image = test_set_algo(image, svm_model)
        else:
            x_test, y_test, image, hog_image = test_set_algo_raw(image, svm_model)

        # Collect all features
        x_test_all.append(x_test)
        y_test_all.append(y_test)

        # Print predictions
        img_name = img.split("/")[-1][:-4]

        print("Actual:", img_name)
        print("Predicted:", y_test)

        # Rescale HoG image
        hog_image = hog_image.astype("float64")

        # Add text to image
        actual_text = f"Actual: {img_name}"
        predicted_text = f"Predicted: {y_test}"

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
    print("Finished!")
    return x_test_all, y_test_all


# Create confusion matrix
def generate_cfn_mtx(y_true: list, y_pred: list, title: "str", labels: list) -> None:
    class_names = ["Negative", "Positive"]
    graph_text = [["TN", "FP"], ["FN", "TP"]]

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Plot
    plt.clf()
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.summer)

    plt.title(f"SVM Confusion Matrix - {title}")
    plt.ylabel("True")
    plt.xlabel("Predicted")

    # Ticks
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(graph_text[i][j]) + " = " + str(cm[i][j]))
    plt.show()


# Print model accuracy and classification
def view_model_acc(y_true: list, y_pred: list) -> None:
    # Accuracy of Model
    acc = accuracy_score(y_true, y_pred)
    print(f"\nModel Accuracy: {acc}\n")

    # Classification Report
    print(
        "\nClassification Report: \n",
        classification_report(y_true, y_pred),
    )


if __name__ == "__main__":
    # Folder name of training data in "inputs" folder
    train_folder = "shapes"

    # Folder name of testing data in "test_images"
    test_folder = "test_cfn"

    # Get feature vectors
    x_train, y_train = generate_hog_features(train_folder)

    # Train SVM model
    svm_model = generate_svm(x_train, y_train)

    # Run SVM over test dataset
    x_test, y_test = run_svm(test_folder, svm_model)
