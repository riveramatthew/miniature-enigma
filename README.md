# Image Classifier Project

## Files Submitted
- All required files are included in the submission package.

---

## Part 1 - Development Notebook

### Package Imports
- All necessary packages and modules are imported in the first cell of the notebook.

### Training Data Augmentation
- Training data is augmented using `torchvision.transforms` with techniques such as random scaling, rotations, mirroring, and cropping.

### Data Normalization
- Training, validation, and testing datasets are properly cropped and normalized.

### Data Loading
- Data for train, validation, and test sets are loaded using `torchvision.datasets.ImageFolder`.

### Data Batching
- Each dataset is loaded into batches using `torch.utils.data.DataLoader`.

### Pretrained Network
- A pretrained network (e.g., VGG16) is loaded from `torchvision.models`.
- The pretrained network parameters are frozen.

### Feedforward Classifier
- A new feedforward classifier network is defined and attached, using the pretrained features as input.

### Training the Network
- Only the feedforward classifier parameters are trained, while pretrained feature parameters remain static.

### Validation Loss and Accuracy
- During training, validation loss and accuracy are computed and displayed.

### Testing Accuracy
- Final accuracy is measured on the test dataset.

### Saving the Model
- The trained model is saved as a checkpoint, including hyperparameters and the `class_to_idx` mapping.

### Loading Checkpoints
- A function is provided that successfully loads a checkpoint and rebuilds the model.

### Image Processing
- The `process_image` function converts a PIL image into a format suitable for model input.

### Class Prediction
- The `predict` function takes an image path and checkpoint, returning the top K most probable classes.

### Sanity Checking with Matplotlib
- A matplotlib figure displays an image alongside its top 5 predicted classes with flower names.

---

## Part 2 - Command Line Application

### Training a Network
- `train.py` trains a new network on a dataset of images and saves the checkpoint.

### Training Validation Log
- Training loss, validation loss, and validation accuracy are printed during training.

### Model Architecture
- Users can choose from at least two pretrained architectures available in `torchvision.models`.

### Model Hyperparameters
- Users can specify hyperparameters such as learning rate, hidden units, and number of epochs via command line options.

### Training with GPU
- Users can enable GPU training with a command line flag.

### Predicting Classes
- `predict.py` reads an image and checkpoint, printing the most likely image class and probability.

### Top K Classes
- The script can output the top K predicted classes with their probabilities.

### Displaying Class Names
- The script supports loading a JSON file to map class indices to real category names.

### Predicting with GPU
- Users can enable GPU usage for inference.

---

Thank you for reviewing this submission!
