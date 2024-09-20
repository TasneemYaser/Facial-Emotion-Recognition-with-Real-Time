#  Facial Emotion Recognition and Real-Time Deployment

## Overview
This project builds and deploys a real-time facial emotion recognition system using deep learning. It is divided into two primary components:
1. **Model Training**: The model is trained on a Kaggle-sourced dataset of facial expressions and implemented using a convolutional neural network (CNN).
2. **Real-Time Detection**: The trained model is applied to detect emotions in real-time using webcam video feed via OpenCV.

## Dataset
The dataset used for this project is a modified version of a FER2013 dataset, containing images categorized into seven different facial expressions. The data is downloaded from Kaggle.

- **Classes**: 7 facial expression categories.
- **Training images**: Located in `images/train`.
- **Validation images**: Located in `images/validation`.

## Model Architecture
The emotion recognition model is a Convolutional Neural Network (CNN) with the following architecture:
- **Input Layer**: Grayscale images resized to 48x48 pixels.
- **Convolutional Layers**: 3 sets of Conv2D layers, each followed by BatchNormalization, ReLU activation, and MaxPooling.
- **Fully Connected Layers**: 3 Dense layers with ReLU activations, and a final Dense layer with softmax activation for classification into 7 emotion categories.
- **Optimizer**: Adam optimizer with a learning rate of 0.0001.
- **Loss Function**: Categorical Crossentropy.

## Training
The model is trained using the Adam optimizer for 30 epochs. Additionally:
- A **learning rate reduction callback** (ReduceLROnPlateau) is used to decrease the learning rate if the validation loss plateaus.
- The dataset is split into training and validation sets for model evaluation.

## Evaluation
To assess the model's performance, the following metrics and visualizations are used:
1. **Training and Validation Loss**: Plotted over 30 epochs to track the learning progress.
2. **Training and Validation Accuracy**: Plotted over 30 epochs to evaluate the model's accuracy.
3. **Confusion Matrix**: Generated after training to visualize the classification performance across the 7 facial expression categories.

## Real-Time Emotion Detection
Once the model is trained and saved as `Finalmodel.h5`, it is loaded into a separate notebook to perform real-time emotion detection using a webcam. The webcam feed is processed frame by frame, and the model predicts the emotion of the detected faces in real time.

