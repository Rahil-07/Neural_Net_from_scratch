# Neural Network from Scratch

This project implements a simple neural network from scratch using Python and NumPy. The neural network is trained to perform binary classification using **gradient descent optimization** and supports various activation functions i.e. **Relu, Tanh, Sigmoid** and regularization techniques **L2 regularization**, **dropout regularization**.

## Dataset

The neural network is trained on the Car vs. Bike Classification Dataset, which can be downloaded from [Kaggle](https://www.kaggle.com/datasets/utkarshsaxenadn/car-vs-bike-classification-dataset). After downloading the dataset, place the downloaded data inside **datasets** folder. you can also use other datasets.

## Project Structure

- `datasets/`: The folder to store the downloaded dataset.
- `dataset_initialization.py`: The Python file to upload images data into array of pixels
- `Neural_Net.py`: The Jupyter Notebook for implementing Nueral Net for deep learning

## Dependencies

- Python 3.7 or higher
- Numpy
- Opencv
- Matplotlib

## Getting Started

To run the neural network and train the model, follow these steps:

1. Clone this repository: git clone https://github.com/Rahil-07/Neural_Net_from_scratch.git
2. Download the dataset and place it in the "datasets" folder.
3. Install the required dependencies:
4. Train the neural network and Adjust hyperparameters such as learning rate, number of epochs, and regularization strength as needed.
5. Evaluate the model's performance:

## Model Architecture

The neural network architecture can be customized by adjusting the `layer_info` list in the `layer_initialization` function. This list specifies the number of neurons in each layer and the activation function for each layer. 

i.e. :

- Input Layer: 12288 neurons (no. of pixel of image)      ##do not include Input Layer in layer_info
- Hidden Layer 1: 16 neurons with ReLU activation
- Hidden Layer 2: 8 neurons with ReLU activation
- Output Layer: 1 neuron with Sigmoid activation (for binary classification)

## Regularization Techniques

The project supports two regularization techniques:

1. L2 Regularization: To enable L2 regularization, set the `lambd` parameter in the training function This helps prevent overfitting by penalizing large weight values.

2. Dropout Regularization: Dropout can be enabled during training by setting the `keep_prob` parameter. Dropout randomly deactivates neurons during each iteration, reducing overreliance on specific neurons and improving generalization.

## Results

During training, the model's cost and accuracy are displayed for every 100 iterations. After training, the final accuracy on both the training and test datasets will be shown.
