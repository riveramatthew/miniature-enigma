Image Classifier Project
Overview
This project implements a deep learning image classifier for flower species using PyTorch and torchvision. It consists of two parts:
Part 1: A Jupyter notebook that develops and trains a convolutional neural network with transfer learning.
Part 2: A command line application with train.py and predict.py scripts that allow users to train a model and predict flower classes from images.
Files Submitted
Jupyter Notebook (HTML format): Contains all model development, training, validation, and testing steps.
train.py: Script to train a new network on a dataset, save checkpoints, and supports various architectures, hyperparameters, and GPU training.
predict.py: Script to predict flower classes from an image using a saved checkpoint; supports top-K predictions, category name mapping, and GPU inference.
Additional helper scripts/files: Utility functions for image processing, checkpoint loading, and class prediction.
cat_to_name.json: Mapping file for class indices to flower names.
Part 1 - Development Notebook
The notebook meets the following criteria:
Package Imports: All necessary packages and modules are imported in the first cell.
Training Data Augmentation: Uses torchvision transforms to augment training data with random scaling, rotations, mirroring, and cropping.
Data Normalization: Training, validation, and testing datasets are cropped and normalized appropriately.
Data Loading and Batching: Uses torchvision’s ImageFolder and DataLoader to load and batch datasets.
Pretrained Network: Loads a pretrained network (e.g., VGG16) with frozen feature parameters.
Feedforward Classifier: Defines a new classifier on top of the pretrained features.
Training the Network: Only the classifier's parameters are trained; the feature extractor is frozen.
Validation Loss and Accuracy: Validation loss and accuracy are displayed during training.
Testing Accuracy: Measures and reports accuracy on the test dataset.
Saving and Loading Checkpoints: Implements functions to save checkpoints with hyperparameters and class mappings and to reload the model from checkpoints.
Image Processing: Includes a process_image function that prepares PIL images for model input.
Class Prediction: Implements a predict function to return top-K class predictions with probabilities.
Sanity Checking: Visualizes an image alongside its top 5 predicted classes and probabilities using matplotlib.
Part 2 - Command Line Application
The command line scripts support the following:
Training a Network: train.py successfully trains a new network on the image dataset.
Training Validation Log: Training loss, validation loss, and validation accuracy are printed throughout training.
Model Architecture: Allows choice from at least two torchvision pretrained architectures (e.g., VGG16, VGG13).
Model Hyperparameters: Supports user-defined learning rate, hidden units, and number of epochs.
GPU Training: Allows training on GPU when specified.
Predicting Classes: predict.py reads an image and checkpoint, then prints the most likely class and probability.
Top K Classes: Supports printing top-K predictions and their probabilities.
Displaying Class Names: Loads a JSON file mapping class indices to flower names.
GPU Prediction: Supports GPU inference during prediction.
