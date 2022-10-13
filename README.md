# RNTN
An implementation of the Recursive Neural Tensor Network (RNTN) described by Socher et al (2013) in "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank"

* What it does: ...
* How to run/install: ...
* Where to get the dataset.
## Gradient derivations

### Notation

 * $d$   - Length of word vector
 * $n$   - Node/layer
 * $x$   - Activation/output of neuron $(x \in \mathbb{R}^{d}$; $\tanh z)$
 * $z$   - Input to neuron $(z \in \mathbb{R}^{d}$; $z = Wx)$
 * $t$   - Target vector $(t \in \mathbb{R}^5$; 0-1 coded)
 * $y$   - Prediction $(y \in \mathbb{R}^5$; output of softmax layer - $softmax(z))$
 * $W_s$ - Classification matrix $(W_s \in \mathbb{R}^{5 \times d})$
 * $W$ - Weight matrix $(W \in \mathbb{R}^{d \times 2d})$
 * $V$ - Weight tensor $(V^{1:d} \in \mathbb{R}^{2d \times 2d \times d} )$
 * $L$ - Word embedding matrix $(L \in \mathbb{R}^{d \times |V|}$, $|V|$ is the size of the vocabulary)
 * $\theta$ - All weight parameters $(\theta = (W_s, W, V, L))$
 * $E$ - The cost as a function of $\theta$
 * $\delta_l$ - Error going to the left child node $(\delta_r$ error to the right child node)

### Softmax
$$y_{i} = \frac{e^{z_i}}{\sum\limits_{j}e^{z_j}}$$

$$\frac{\partial y_i}{\partial z_j} = y_{i}(\delta_{ij} - y_{j})$$

$\delta_{ij}$ is the Kronecker's delta: 
$$
\delta_{ij} = 
  \begin{cases} 
    0 &\text{if } i \neq j, \\ 
    1 &\text{if } i=j. 
  \end{cases}
$$

  
* Reference: R. Socher, A. Perelygin, J.Y. Wu, J. Chuang, C.D. Manning, A.Y. Ng and C. Potts. 2013. Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank. In EMNLP.
