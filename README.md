
# Tomato Leaf Disease Identifier

Welcome to the **Tomato Leaf Disease Identifier** project! This repository contains the code and resources for training a deep learning model that identifies diseases in tomato leaves using image classification. The model is built and trained using TensorFlow, and the goal is to provide an accurate tool for diagnosing tomato leaf diseases based on leaf images.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Results](#training-results)
- [Graph Analysis](#graph-analysis)
- [How to Use](#how-to-use)
- [Installation](#installation)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
Tomato leaf diseases can have a significant impact on crop yield and quality. This project aims to leverage machine learning to automate the identification of diseases from images of tomato leaves. By doing so, farmers and agricultural professionals can detect diseases early and take timely action.

## Dataset
The dataset used for training consists of labeled images of tomato leaves with various diseases, as well as healthy leaves. It was sourced from publicly available datasets focused on plant pathology.

### Classes
- Healthy leaves
- Diseased leaves (specific diseases vary depending on the dataset)

## Model Architecture
The model is based on a convolutional neural network (CNN) architecture. Key features of the architecture include:
- **Input Layer**: Preprocessed images resized to a standard dimension.
- **Convolutional Layers**: Extract spatial features from the images.
- **Pooling Layers**: Downsample feature maps to reduce computational complexity.
- **Fully Connected Layers**: Combine extracted features for classification.
- **Output Layer**: Softmax activation for multi-class classification.

## Training Results
The model was trained for 50 epochs using TensorFlow. Below is a detailed analysis of the training and validation accuracy and loss.

### Graph Analysis
#### Training and Validation Accuracy
The graph below shows how the training and validation accuracy evolved over the epochs. The training accuracy steadily increased, reaching close to 1.0, indicating that the model learned the patterns in the training data effectively. However, the validation accuracy exhibited more variability, suggesting potential overfitting or challenges with generalization.

#### Training and Validation Loss
The graph below illustrates the training and validation loss. The training loss decreased consistently, demonstrating that the model minimized the error on the training set. The validation loss, on the other hand, fluctuated significantly, which might indicate overfitting or noise in the validation data.

![Training and Validation Graphs](./Figure_2.png)

### Insights:
- The training process shows strong learning on the training set, as indicated by high accuracy and low loss.
- Validation accuracy and loss fluctuations suggest further steps, such as data augmentation or regularization, might improve the model’s performance.

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/RidwanSiddique/TomatoLeafDiseaseIdentifier.git
   ```
2. Navigate to the project directory:
   ```bash
   cd TomatoLeafDiseaseIdentifier
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the training script:
   ```bash
   python train_model.py
   ```
5. Use the trained model for inference:
   ```bash
   python predict.py --image path/to/image.jpg
   ```

## Installation
Ensure you have Python 3.8 or above installed. Use the provided `requirements.txt` file to install necessary dependencies:
```bash
pip install -r requirements.txt
```

## Future Improvements
- Add more robust data augmentation techniques.
- Experiment with different CNN architectures.
- Implement transfer learning with pre-trained models like ResNet or EfficientNet.
- Perform hyperparameter tuning for optimization.
- Address validation loss fluctuations through regularization or additional dataset preprocessing.

## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request if you have suggestions or improvements.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
Thank you for exploring the **Tomato Leaf Disease Identifier** project! If you have any questions or feedback, feel free to reach out.
