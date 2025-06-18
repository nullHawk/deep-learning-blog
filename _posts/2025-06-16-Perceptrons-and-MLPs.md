---
title: Perceptron and MLPs
date: 2025-06-16
categories: [Deep Learning]
tags: [deep-learning, neural-networks]
author: nullHawk
math: true
description:
---

Ever wondered how a computer can learn to tell apples from oranges? At the heart of it there is something deceptevly simple - the **Perceptron**.

In this post, we'll discuss about Perceptron and MLPs (Multi-layer Perceptrons) - ideas that laid foundation for modern Deep Learning.

## Inspiration

Perceptron's roots lie in a 1943 paper by Warren McCulloch and Walter Pitts, where they pruposed a simple model of biological neuron. The neuron's cell body recieves input via its treelike projections, called dendrites. The cell body performs some computation on these inputs. Then, based on the results of that computation, it may send an electrical signal spiking along another, longer projection, called the axon. That signal travels along the axon and reaches its branching terminals, where it's communicated to the dedrites of neighboring neurons. And so it goes. Neurons interconnected in this manner form a biological netowrk.
![Biological neuron structure](https://i.ibb.co/F459kq5C/biological-neuron.jpg)
*Figure: Structure of a biological neuron showing dendrites, cell body, axon, and synaptic terminals.*
McCulloch and Pitts turned this into a simple computational model, and artificial neuron. They showed how by using one such artifical neoron, or neurode(for "neuron" + "node"), one could implement certain basic Boolean logical operations such as AND, OR, NOT and so on, which are building blocks of digital computation.
![McCulloch-Pitts model](https://i.ibb.co/rGkK9VMZ/neurode.png)
*Figure: Simple version of McCulloch-Pitts model of neurode*
In neurode, $x_1$ and $x_2$ can be either 0 or 1. In formal notation, we can say:

$$
x_1, x_2 \in \{0,1\}
$$

Where the nerode takes these two inputs and sums it and if the sum is greater than or equals to $\theta$ then outputs 1 or else it gives output 0. Mathematically we can say that if given

$$
g(x) = x_1 + x_2 + x_3 + ... + x_n = \sum_{i=1}^{n} x_i
$$
<br>
and
<br>
$$
f(z) = 
\begin{cases}
0, &  z < \theta \\
1, &  z \geq \theta
\end{cases}
$$
Then what perceptron does is
<br>
$$
y = f(g(x)) = \begin{cases}
0, & \text{if } g(x) < \theta \\
1, & \text{if } z \geq \theta
\end{cases}
$$

MCP(McCulloch-Pitts) model amazing and yet limited, you can use combinations of it to create any type of Boolean logic and yet, the upshot was simply machine that could compute, not learn. In particular, the value of $\theta$ had to be hand-engineered; the neuron couldn't examin the data and figure out $\theta$.
It's no wonder Rosenblatt's perceptron made such a splash. It could learn its weights from data. The weights encoded some knowledge, however minimal, about the patterns in the data and remembered them, in a manner of speaking.

[For code implemenation of neurode you can refer here :)](https://github.com/nullHawk/deep_learning/blob/main/perceptron%26mlp/mcp_neurode.py)

## Perceptron
While McCulloch and Pitts had developed models of the neuron, networks of these artificial neurons could not learn. In the context of biological neurons, hebb had proposed a mechanism for learning that is often succinctly, but somewhat erroneously, put as "**Neurons that fire together wire together**". More precisely, according to this way of thinking, our brains learn because connections between neurons strengthen when one neuron's ouput is consistently involved in the firing of another, and they weaken when this it not so. The process is called **Hebbian Learning**. It was Rosenblatt who took the work of these pioneers and synthetsized it into a new idea: artificial neurons that reconfigure as they learn, embodying information in the strength of their connections.

![Perceptron model](https://i.ibb.co/8n6XBdGY/perceptron.png)
*Figure: Basic perceptron model with weighted inputs and threshold activation*

But what exactly is a perceptron, and how does it learn? In its simplest form, a perceptron is an augmented McCulloch-Pitts neuron imbued with a learning algorithm. What follows is an example with two inputs. Note that each input is being multiplied by its corresponding weight.
The computation carried out by the perceptron goes like this:

$$
g(x) = w_1x_1 + w_2x_2 + ... + w_nx_n + b = \sum_{i=1}^{n} w_ix_i + b
$$

$$
f(z) = 
\begin{cases}
1, &  z > 0 \\
-1, & z \leq 0
\end{cases}
$$

$$
y = f(g(x)) = 
\begin{cases}
1, &  g(x) > 0 \\
-1, & g(x) \leq 0
\end{cases}
$$

The main difference from the MCP model presented earlier is that the perceptron's input don't have binary (0 or 1), but can take on any value. Also these inputs are multiplied b their corresponding wights, so we now have a weighted sum. Added to that is an additional term b, the bias.

### Training a Perceptron
- **Step 1:** Initialise weight vector to zero
- **Step 2:** for each data point $x$ in the training dataset, do the following

    $$
    \begin{aligned}
    \text{if prediction} \neq y: \\
    w_{\text{new}} &= w_{\text{old}} + \eta \cdot y \cdot x \\
    b_{\text{new}} &= b_{\text{old}} + \eta \cdot y
    \end{aligned}
    $$
- **Step 3:** If there were no updates to the weight vector in Step 2, terminate other wise go to Step 2 and itrate overall the datapoints again.

You can find a code implementation of perceptron in [this repository](https://github.com/nullHawk/deep_learning/blob/main/perceptron%26mlp/perceptron.py)

### Perceptron Convergenge Proof
The perceptron convergence theorem states that if the training data is linearly separable, the perceptron learning algorithm will converge to a solution in a finite number of steps. This means that it will find a set of weights and bias that correctly classifies all training examples.
$$
\begin{aligned}
\text{Let } w \text{ be the weight vector and } b \text{ be the bias.} \\
\text{final bound} : t \leq \frac{R^2}{\gamma^2} \\
\text{where } \gamma \text{ is the minimum distance from the dataset point to the ideal decision boundary} \\
\end{aligned}
$$

You can read more about the perceptron convergence theorem in [this paper](https://www.cs.columbia.edu/~mcollins/courses/6998-2012/notes/perc.converge.pdf).

## Limitations of Perceptron
- The perceptron can only learn linearly separable functions.
- It cannot learn non-linear functions, such as XOR.
- It can only output binary values (0 or 1), which limits its applicability to binary classification problems.
- It is sensitive to the choice of learning rate, which can affect convergence.

## Sigmoid neurons
The perceptron is a powerful model, but it has its limitations. It can only output binary values (0 or 1). This means that it can only be used for binary classification problems. However, many real-world problems require a model that can output continuous values. For example, in regression problems, we want to predict a continuous value, such as the price of a house or the temperature of a city.
To address this limitation, the sigmoid neuron was introduced. The sigmoid neuron is a perceptron that uses a sigmoid activation function instead of a step function. The sigmoid function is defined as:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Advantages of using sigmoid activation function:
- It can output continuous values between 0 and 1, making it suitable for regression problems
- It is differentiable, which allows for the use of gradient descent to optimize the weights and bias
- It has a smooth gradient, which helps to avoid the vanishing gradient problem
- It is used in multi-layer perceptrons (MLPs)

## Multi-layer Perceptrons (MLPs)
Multi-layer perceptrons (MLPs) are a type of neural network that consists of multiple layers of neurons. Each layer is fully connected to the next layer, and each neuron in a layer receives input from all neurons in the previous layer. MLPs can be used for both classification and regression problems.

![MLP](https://i.ibb.co/TxfGtH5d/MLP.png)
*Figure: Structure of a Multi-layer Perceptron (MLP) with input, hidden, and output layers.*

MLPs are composed of an input layer, one or more hidden layers, and an output layer. The input layer receives the input data, the hidden layers perform computations on the data, and the output layer produces the final output.

### Working of MLPs
Let's look at the key mechanisms of MLPs I will be discussing these in later blog posts in detail, but for now, here's a brief overview:

1. **Forward Propagation**: The input data is passed through the network, layer by layer, until it reaches the output layer. Each neuron applies a weighted sum of its inputs, adds a bias term, and passes the result through an activation function.
    - Weighted Sum: Each neuron computes a weighted sum of its inputs $z = \sum_{i} w_i x_i + b$
    - Activation Function: Calculated $z$ is passed through activation function. Some of the commonly used activation functions are:
        - Sigmoid($\sigma$): $\frac{1}{1 + e^{-z}}$
        - ReLU(Rectified Linear Unit): $f(z) = \max(0, z)$
        - Tanh: $tanh(z) = \frac{2}{1 + e^{-2z}} - 1$
2. **Loss Function**: The output of the MLP is compared to the true labels using a loss function, which calculates the error between the predicted output and the true output. Common loss functions are:
    - Mean Squared Error (MSE) for regression tasks: 
      $$
      \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
      $$
    - Cross-Entropy Loss for classification tasks:
      $$
      \text{Cross-Entropy} = -\frac{1}{n}\sum_{i=1}^{n} y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)
      $$
3. **Backpropagation**: Backpropagation is the process of calculating the gradients of the loss function with respect to the weights and biases of the MLP. This is done by applying the chain rule of calculus to propagate the error backward through the network. 

   - For each neuron, the gradient of the loss with respect to the output is calculated:
   $$
   \frac{\partial L}{\partial \hat{y}} = \hat{y} - y
   $$
   where $\hat{y}$ is the predicted output and $y$ is the true label.
   
   - The gradient of the output with respect to the weighted sum $z$ is calculated using the derivative of the activation function:
   $$
   \frac{\partial \hat{y}}{\partial z} = \sigma'(z) = \sigma(z)(1 - \sigma(z))
   $$
   
   - The gradient of the weighted sum with respect to each weight $w_i$ is calculated:
   $$
   \frac{\partial z}{\partial w_i} = x_i
   $$
   
   - Finally, the gradient of the loss with respect to each weight is calculated using the chain rule:
$$
\frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w_i}
$$

4. **Optimization**:
The weights and biases of the MLP are updated using an optimization algorithm, such as stochastic gradient descent (SGD) or Adam. The goal is to minimize the loss function by adjusting the weights and biases in the direction of the negative gradient. $\eta$ is the learning rate and $L$ is the loss function.

$$
w_i = w_i - \eta \cdot \frac{\partial L}{\partial w_i}
$$

$$
b = b - \eta \cdot \frac{\partial L}{\partial b}
$$

Though I have summarized the these topics here but they need a detailed explanation. I will be writing separate blog posts on these topics in the future, so stay tuned!

You can find the implementation of MLP in [this repository](https://github.com/nullHawk/deep_learning/blob/main/perceptron%26mlp/mlp.py)

## Conclusion
In this post, we explored the origins of perceptrons and how they evolved into multi-layer perceptrons (MLPs). We discussed the basic structure of perceptrons, their training process, and the limitations of single-layer perceptrons. We also introduced the concept of sigmoid neurons and how they paved the way for MLPs, which can learn complex patterns in data.
MLPs are a powerful tool in deep learning, enabling us to tackle a wide range of problems, from image classification to natural language processing. As we continue our journey into deep learning, understanding the foundations of perceptrons and MLPs will be crucial for grasping more advanced concepts and architectures in the field. 


