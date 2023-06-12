# Exploring-MultiLayer-Perceptrons

If you are unable to render the notebook on github, you can view it [here](https://nbviewer.org/github/akash18tripathi/Exploring-MultiLayer-Perceptrons/blob/main-Exploring%20MultiLayer%20Perceptrons.ipynb)

## Introduction
In this Notebook, we will be working with the MNIST dataset to explore the importance of different Multi-Layer Perceptron (MLP) components. The MNIST dataset consists of 70,000 handwritten digit images, each of which is 28x28 pixels in size. Our goal is to use an MLP to classify these images into one of 10 categories (0-9).

To improve the performance of our MLP, we will experiment with various techniques such as **Dropout**, **Batch Normalization**, **Loss Functions**, **Stochastic batch and mini-batch gradient descent**, and more. Please note, we are using mini-batch unless explicitly specified.

In addition, we will experiment with different optimization algorithms such as **Stochastic Gradient Descent**, **Adam**, and **RMSprop** to find the optimal weights and biases for our MLP. We will use stochastic batch and mini-batch gradient descent, which involve updating the weights and biases of the network based on a small batch of randomly sampled training examples, to speed up the training process and reduce memory usage.

By the end of these experiments, we will have gained a deeper understanding of the various components that make up an MLP and their importance in achieving high performance in classification tasks. We will have gained hands-on experience in experimenting with these components and learned how to fine-tune an MLP to achieve the best possible performance on the MNIST dataset.

## Concepts Covered
- MNIST dataset
- Multi-Layer Perceptron (MLP)
- Dropout
- Batch Normalization
- Loss Functions
- Stochastic Gradient Descent
- Mini-batch Gradient Descent
- Optimization Algorithms (Stochastic Gradient Descent, Adam, RMSprop)

## Repository Structure
- `README.md`: Overview and instructions for the repository
- `Exploring MultiLayer Perceptrons.ipynb`: Jupyter Notebook containing the code and experiments

## Dropout
Dropout is a regularization technique used in neural networks to prevent overfitting. It randomly sets a fraction of the input units to 0 at each training update, which helps to reduce the reliance of the network on specific features and encourages the network to learn more robust representations.

## Batch Normalization
Batch Normalization is a technique used to normalize the activations of each layer in a neural network. It helps to stabilize the learning process by reducing the internal covariate shift and enables faster convergence and better generalization.

## Loss Functions
Loss functions quantify the error or discrepancy between the predicted output and the actual target output of a neural network. Different loss functions are used for different types of tasks such as classification, regression, and more. The choice of an appropriate loss function is crucial for training the network effectively.

## Stochastic Gradient Descent
Stochastic Gradient Descent (SGD) is an optimization algorithm commonly used for training neural networks. It updates the weights and biases of the network based on the gradient of the loss function computed on a single training example at each iteration. SGD is computationally efficient but may exhibit more noisy convergence compared to other optimization algorithms.

## Mini-batch Gradient Descent
Mini-batch Gradient Descent is a variant of stochastic gradient descent that updates the weights and biases of the network based on a small batch of randomly sampled training examples instead of a single example. It strikes a balance between the computational efficiency of stochastic gradient descent and the stability of batch gradient descent.

## Optimization Algorithms
Optimization algorithms play a crucial role in training neural networks. Stochastic Gradient Descent (SGD) is a popular optimization algorithm, but it can be slow to converge. To overcome this, we will also explore other optimization algorithms such as **Adam** and **RMSprop**. These algorithms adaptively adjust the learning rate for each parameter based on the first and second moments of the gradients, which can lead to faster convergence and better performance.

## Libraries used

The MLP is trained using Pytorch library but with using Custom MLP.

## Contributing
Contributions are welcome and encouraged!
