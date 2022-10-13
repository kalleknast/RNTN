# RNTN
An implementation of the Recursive Neural Tensor Network (RNTN) described by Socher et al (2013) in "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank"

* What it does: ...
* How to run/install: ...
* Where to get the dataset.
## Gradient derivations

### Softmax
$$
y_{i} = \frac{e^{z_i}}{\sum\limits_{j}e^{z_j}}

\frac{\partial y_i}{\partial z_j} = y_{i}(\delta_{ij} - y_{j})
$$

$$\delta_{ij}$$ is the Kronecker's delta:
	$$ \delta_{ij} = \begin{cases}
	0 &\text{if } i \neq j,   \\
	1 &\text{if } i=j.   \end{cases} $$
  
  
* Reference: R. Socher, A. Perelygin, J.Y. Wu, J. Chuang, C.D. Manning, A.Y. Ng and C. Potts. 2013. Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank. In EMNLP.
