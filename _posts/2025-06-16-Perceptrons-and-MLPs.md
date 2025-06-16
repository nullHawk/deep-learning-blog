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
-1, & z leq 0
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

This article is still in progress...


