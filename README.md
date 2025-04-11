# CIFAR-10 Image Classification using Transfer Learning with ResNet50

This repository contains a Jupyter Notebook (`Transfer_learning_on_CFAR_IMAGE.ipynb`) demonstrating how to perform transfer learning using a pre-trained ResNet50 model to classify images from the CIFAR-10 dataset.

## Description

The notebook walks through the process of leveraging a deep learning model (ResNet50), originally trained on the large ImageNet dataset, and adapting it for a different task: classifying the smaller, lower-resolution images of the CIFAR-10 dataset.

## Dataset

The **CIFAR-10** dataset is used. It consists of 60,000 32x32 color images distributed across 10 distinct classes:
`['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']`

The dataset is split into 50,000 training images and 10,000 validation (test) images.

## Model & Technique

The core technique employed is **transfer learning**.

-   **Base Model:** ResNet50, pre-trained on the ImageNet dataset, is utilized as a fixed feature extractor. The original classification layer (`include_top=False`) is removed.
-   **Input Adaptation:** Since ResNet50 expects 224x224 input images, the 32x32 CIFAR-10 images are first upsampled using `tf.keras.layers.UpSampling2D(size=(7, 7))`.
-   **Preprocessing:** Images are preprocessed using `tf.keras.applications.resnet50.preprocess_input` to match the format ResNet50 was trained on.
-   **Custom Classifier:** A new classification head is added on top of the ResNet50 base. This head consists of:
    -   `GlobalAveragePooling2D`
    -   `Flatten`
    -   `Dense(1024, activation='relu')`
    -   `Dense(512, activation='relu')`
    -   `Dense(10, activation='softmax')` (for the 10 CIFAR-10 classes)

## Workflow in the Notebook

1.  **Imports:** Necessary libraries like TensorFlow, Keras, NumPy, Matplotlib, and PIL are imported.
2.  **Constants & Helpers:** Define class names, batch size, and helper functions for displaying images and plotting training metrics.
3.  **Data Loading:** Load the CIFAR-10 dataset directly using `tf.keras.datasets.cifar10.load_data()`.
4.  **Data Visualization:** Display sample images from the training and validation sets.
5.  **Preprocessing:** Apply the ResNet50-specific preprocessing to the image data.
6.  **Model Definition:** Define the transfer learning model architecture by combining the upsampling layer, ResNet50 base (feature extractor), and the custom classifier head.
7.  **Compilation:** Compile the Keras model using the 'SGD' optimizer, 'sparse_categorical_crossentropy' loss function, and 'accuracy' metric.
8.  **Training:** Train the model for 4 epochs using the preprocessed training data and validate on the preprocessed validation data. The training history (loss, accuracy) is recorded.
9.  **Evaluation:** Evaluate the final model's performance on the validation set.
10. **Metrics Visualization:** Plot the loss and accuracy curves for both training and validation phases over the epochs.
11. **Prediction Visualization:** Generate predictions on the validation set and display sample images alongside their predicted labels.

## Requirements

The primary libraries needed to run this notebook are:
-   TensorFlow (>= 2.x)
-   NumPy
-   Matplotlib
-   Pillow (PIL)


After 4 epochs of training, the model achieves approximately **94.5% accuracy** on the CIFAR-10 validation set, demonstrating the effectiveness of transfer learning even with significant differences in image resolution between the source (ImageNet) and target (CIFAR-10) datasets.
