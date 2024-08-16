# AI Art vs Human Art Binary Classifier

This project is a binary classifier designed to differentiate between AI-generated art and human-created art. The model leverages deep learning techniques to analyze images and predict their origin—whether they were created by an AI or a human.

## Project Overview

The project involves the following key steps:
1. **Dataset Preparation**: 
    - Two categories of images are collected: AI-generated art and human-created art.
    - The images are organized into training and validation datasets with a 90-10 split.

2. **Model Development**:
    - A Convolutional Neural Network (CNN) model is trained to classify the images.
    - The model undergoes initial training, followed by fine-tuning to improve accuracy.

3. **Model Evaluation**:
    - The model's performance is evaluated using accuracy and loss metrics for both training and validation datasets.
    - Fine-tuning helps to enhance the model's predictive power.

4. **Model Deployment**:
    - The final model is saved in `.h5` format for further use in applications or for deployment.

## Installation and Setup

To set up and run this project locally, follow these steps:

### Prerequisites

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- OpenCV

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Malhar-Girgaonkar/AI_ART_Classifier.git
    cd AI_ART_Classifier
    ```
2. Make sure to have proper kernal and python modules required for training model pre installed.

### Dataset Information

1. The dataset used comprises of multiple image formats like `.jpeg` , `.png` , `.jpg` ,etc.
   
2. The dataset used was a customized mix of data from multiple datasets available online out of which two prominant datasets were [WikiART dataset](https://www.kaggle.com/datasets/ipythonx/wikiart-gangogh-creating-art-gan) by Innat and [AI recognition dataset](https://www.kaggle.com/datasets/0b54724023dc63c75bce5334e774f4bbac561120ab815a63013efc4558cb51fa) by Nathan Koliha on Kaggle.

3. The final dataset created is 22GB worth of images split into a specified directory structure.

4. The function `create_human_dataset(human_art_dir, ai_images_count, output_dir)` Creates final form of dataset made by combining equal parts of data from above two datasets.

5. The function `split_data((SOURCE, TRAINING, VALIDATION, SPLIT_SIZE)` basically takes dataset and splits it into training and validation sets.

### Dataset Structure

The dataset should be organized in the following structure:

```
AI_Identifier/
├───Dataset_used
│   ├───Training_Dataset
│   │   ├───AI
│   │   └───Human
│   └───Validation_dataset
│       ├───AI
│       └───Human
|
├───Main_dataset
│   ├───AI
│   └───Human
|
├───Fine_tuning_models
|   |__finetune_model_V1.keras
|
├───Fine_tuning_models_for_app
|   |__finetune_model_V1.h5
|
├───Model_Saves
|   |__AI_Human_Art.h5
|
├───Model_Saves_keras
|   |__AI_Human_Art.keras
|
└───__pycache__
```

### Running the Project

1. **Create the Dataset**:
   - The dataset is generated and split into training and validation sets by running the relevant cells in the notebook.

2. **Train the Model**:
   - The CNN model is trained on the prepared dataset. Training is performed using a script or by running the cells in the notebook.

3. **Evaluate the Model**:
   - After training, evaluate the model's performance using the validation dataset. Metrics like accuracy and loss are plotted for analysis.

4. **Test the Model**:
   - The trained model can be tested on new images to predict whether they are AI-generated or human-created.

### Fine-Tuning and Saving the Model

- Fine-tuning is implemented to improve the model's performance by adjusting the learning rate and training for additional epochs.
- The final model is saved as `finetune_model.h5` for future use.

### Plotting Results

- Accuracy and loss curves are plotted to visualize the model's performance during training and validation.

## Usage

You can load the trained model and use it to predict the origin of any new art pieces:

```python
from tensorflow.keras.models import load_model

# Load the model
model = load_model('Fine_tuning_models/finetune_model.h5')

# Predict the class of a new image
image_path = 'path_to_your_image.jpg'
result = preprocess(image_path)
```

## Contributing
Contributions are welcome! If you have suggestions for improvements or new features, feel free to create a pull request or open an issue.

## License
This project is licensed under the MIT License.
