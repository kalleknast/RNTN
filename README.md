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

### Cost function $E$
$$
		E(\theta) = - \sum\limits_{i}\sum\limits_{j}{t_{j}^{i} \log{y_{j}^{i}} + \lambda||\theta||^2}
$$
$$
		\frac{\partial E}{\partial y_j} = \frac{t_j}{y_j}
$$

### Activation function

$$
		x_i = \tanh{z_i}
$$
$$
		\frac{\partial x_i}{\partial z_i} = 1 - \tanh^2{z_i}
$$

### Derivative of $E$ w.r.t. the sentiment classification matrix $W_s$

$$
		\frac{\partial E}{\partial W_s} = 
		\sum\limits_{k}\frac{\partial E}{\partial y_k}{\frac{\partial y_k}{\partial z^{s}}}{\frac{\partial z^{s}}{\partial W_{s}}}
$$

#### Derivative of the cost function:
$$
			\frac{\partial E}{\partial y} = \frac{t}{y}
$$
#### Derivative of the $softmax$ function:
$$
			\frac{\partial y_k}{\partial z^{s}_{i}} = y_{i}(\delta_{ik} - y_{k})
$$
#### Derivative of the input:
$$
			\frac{\partial z^{s}}{\partial W_s} = x
$$
#### Combined:
$$
			\begin{split}
				\frac{\partial E}{\partial W_s}
				= \sum\limits_{k}\frac{t_k}{y_k}y_{k}(\delta_{ik} - y_{i})x_j \\
			  = x_j \sum\limits_{k}{t_k (\delta_{ik}-y_i)} \\
			  = x_j(y_i - t_i)
			\end{split}
$$
### Derivative of $E$ w.r.t. the weight matrix $W$
#### For one training sentence:
$$
		\frac{\partial E}{\partial W} = 
		\sum\limits_{k}\frac{\partial E}{\partial y_k}
		\frac{\partial y_k}{\partial z_{s}}
		\frac{\partial z_{s}}{\partial x}
		\frac{\partial x}{\partial z}
		\frac{\partial z}{\partial W}
$$
#### Derivative of input to $node_n$ w.r.t. activation of $node_{n-1}$:
$$
		\frac{\partial z}{\partial x} = W
$$
#### Derivative of a node's activation w.r.t. its input:
$$
		\begin{split}
			\frac{\partial x}{\partial z} = 1 - \tanh^2z \\
			f'(x) = 1 - x^2 \\
			f' \bigg( \bigg[ \begin{array}{c} x^l \\ x^r \end{array} \bigg] \bigg) = 
			1 - \bigg[ \begin{array}{c} x^l \\ x^r \end{array} \bigg] \otimes \bigg[ \begin{array}{c} x^l \\ x^r \end{array} \bigg]
		\end{split}
$$
#### Derivative of a node's input w.r.t. its weight matrix $W$:
$$
		\frac{\partial z}{\partial W} = x
$$
#### Combined:
$$
		\begin{split}
			\delta^s = W_s{^T}(y - t) \otimes f'(x_n) \\
			\frac{\partial E}{\partial W} = W^T \delta^s
			\otimes f' \bigg( \bigg[ \begin{array}{c} x_{n-1}^l \\ x_{n-1}^r \end{array} \bigg] \bigg) \bigg[  \begin{array}{c} x_{n-1}^l \\ x^{r}{_{n-1}} \end{array} \bigg]^T\\
		\end{split}
$$

### Derivative of $E$ w.r.t. the slice $k$ of the tensor layer $V^{[k]}$

#### Top node $(node_n)$:
$$
			\begin{split}
				\delta^s = W_s{^T}(y - t) \otimes (1 - x{_n}^2) \\
				\frac{\partial E_n}{\partial V^{[k]}} = 
				\delta^s{_k} \bigg[ \begin{array}{c} x^l{_{n-1}} \\ x^r{_{n-1}} \end{array} \bigg]	
				\bigg[  \begin{array}{c} x_{n-1}^l \\ x^{r}{_{n-1}} \end{array} \bigg]^T \\
			\end{split}
$$
#### Left child node $(node_{n-1})$:
$$
			\begin{split}
				\delta_{n} = \delta^{s,n} \\
				\delta^{n-1}_{k} = \big( W^T \delta^n + S \big) 
				\otimes f' \bigg( \bigg[ \begin{array}{c} x^l_{n-1}\\ x^r_{n-1} \end{array} \bigg] \bigg) \\
				S = \sum\limits_{k = 1}^d \delta^n \bigg( V^{[k]} + \big(V^{[k]})^T \bigg) \bigg[ \begin{array}{c} x^l_{n-1}\\ x^r_{n-1} \end{array} \bigg] \\
				\delta^{n-1}_l = \delta_l^{s,n-1} + \delta^{n-1}[1:d] \\
				\frac{\partial E_{n-1}}{\partial V^{[k]}} = 
				\frac{\partial E_n}{\partial V^{[k]}} + \delta^{n-1}_l \bigg[  \begin{array}{c} x_{n-2}^l \\ x^{r}{_{n-2}} \end{array} \bigg]^T
			\end{split}
$$
  
* Reference: R. Socher, A. Perelygin, J.Y. Wu, J. Chuang, C.D. Manning, A.Y. Ng and C. Potts. 2013. Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank. In EMNLP.
