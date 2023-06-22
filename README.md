# Apple Disease Detection using Apple Leaves as Dataset

This repository contains the code and resources for the project "Apple Disease Detection using Apple Leaves as Dataset". The project aims to develop a machine learning-based system to automatically detect and classify various diseases affecting apple trees by analyzing images of apple leaves.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Apple diseases pose a significant threat to apple production, leading to substantial crop losses. Traditional methods of disease detection are time-consuming and labor-intensive. This project aims to automate the process using machine learning and computer vision techniques. By training a model on a dataset of apple leaf images exhibiting various disease symptoms, we can accurately identify and classify different apple diseases.

## Installation
1. Clone the repository: `git clone https://github.com/pranavsr10/Apple-Disease-Detection.git`
2. Install the required dependencies: `pip install -r requirements.txt`

## Usage
1. Preprocess the dataset by following the instructions in the `Apple_disease_detection.ipynb` notebook.
2. Train the disease detection model using the `Apple_disease_detection.ipynb` notebook.
3. Evaluate the trained model's performance using the `Apple_disease_detection.ipynb` notebook.
4. Use the trained model to predict diseases in new apple leaf images with the `prediction.ipynb` notebook.

## Dataset
The dataset used in this project consists of a collection of apple leaf images, each labeled with the corresponding disease category. The dataset is available at [link to dataset](https://www.kaggle.com/datasets/ludehsar/apple-disease-dataset).

## Model Training
The model is trained using a convolutional neural network (CNN) architecture. The implementation utilizes popular deep learning libraries such as TensorFlow and Keras. The training process involves feeding the preprocessed apple leaf images into the CNN model and optimizing it using appropriate loss functions and optimization algorithms.

## Evaluation
The performance of the disease detection model is evaluated using various metrics such as accuracy, precision, recall, and F1-score. These metrics provide insights into the model's ability to correctly classify apple leaf images into their respective disease categories. The evaluation results can be found in the `evaluation.ipynb` notebook.

## Future Improvements
Some potential areas for future improvement of the project include:
- Increasing the size and diversity of the dataset to improve the model's generalization.
- Exploring advanced deep learning techniques, such as transfer learning, to enhance the model's performance.
- Implementing real-time disease detection using edge computing or IoT devices.
- Refining the user interface for easy interaction with the disease detection system.


## License
This project is licensed under the [MIT License](LICENSE).

