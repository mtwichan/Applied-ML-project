import os
from typing import Union

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def generate_hog_features(folder_name: str) -> Union[list, list]:
    # Get HOG Features from Training Set
    img_path = folder_name
    img_path = f"input/{img_path}"

    x_test = []
    y_test = []

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

            # Read image and resize to 128 x 256 - HoGs requires 1:2 ratio
            image = cv2.imread(image_path)
            image = cv2.resize(image, (128, 256))

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
            x_test.append(hog_desc)
            y_test.append(path)

    return x_test, y_test


def generate_svm(x_test: list, y_test: list) -> LinearSVC:
    svm_model = LinearSVC(random_state=42, tol=1e-5)
    svm_model.fit(x_test, y_test)
    return svm_model


def generate_cfn_mtx(y_test: list, y_pred: list) -> None:
    cm = confusion_matrix(y_test[: len(y_pred)], y_pred, labels=["squares", "blank"])

    plt.clf()
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.summer)
    classNames = ["Negative", "Positive"]

    plt.title("SVM Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")

    # Ticks
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames)
    plt.yticks(tick_marks, classNames)
    graph_text = [["TN", "FP"], ["FN", "TP"]]

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(graph_text[i][j]) + " = " + str(cm[i][j]))
    plt.show()


def run_svm(folder_name: str, svm_model: LinearSVC) -> Union[list, list]:
    # Run SVM model on test images
    print("Testing SVM model on testing set ...")

    img_path = folder_name
    img_path = f"test_images/{img_path}"
    img_paths = os.listdir(img_path)

    output_path = f"outputs/"

    x_pred_all = []
    y_pred_all = []

    # loop over the test dataset folders
    for (i, img) in enumerate(img_paths):
        # File pathing
        img = f"{img_path}/{img}"
        image = cv2.imread(img)

        # Resize to 128 x 256 - HoGs requires 1:2 ratio
        resized_image = cv2.resize(image, (128, 256))

        # Generate HoG for image
        (hog_desc, hog_image) = feature.hog(
            resized_image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            transform_sqrt=True,
            block_norm="L2",
            visualize=True,
        )

        # Run SVM prediction
        x_pred = hog_desc.reshape(1, -1)
        y_pred = svm_model.predict(x_pred)[0]

        x_pred_all.append(x_pred)
        y_pred_all.append(y_pred)

        # Print predictions
        img_name = img.split("/")[-1][:-4]

        print("Actual:", img_name)
        print("Predicted:", y_pred)

        # Rescale HoG image
        hog_image = hog_image.astype("float64")

        # Add text to image
        actual_text = f"Actual: {img_name}"
        predicted_text = f"Predicted: {y_pred}"

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

    # Accuracy of Model
    acc = accuracy_score(y_test[: len(y_pred_all)], y_pred_all)
    print(f"\nModel Accuracy: {acc}\n")

    # Classification Report
    print(
        "\nClassification Report: \n",
        classification_report(y_test[: len(y_pred_all)], y_pred_all),
    )

    return x_pred_all, y_pred_all


if __name__ == "__main__":
    # Point to folder
    folder_name = "shapes"

    # Get x and y test feature data
    x_test, y_test = generate_hog_features(folder_name)

    # Create SVM and train data
    svm_model = generate_svm(x_test, y_test)

    # Get x and y predicted data
    x_pred, y_pred = run_svm(folder_name, svm_model)

    # Create confusion matrix
    generate_cfn_mtx(y_test, y_pred)
