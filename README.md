# energy-min-neural-networks

use various regularization techniques to minimize energy in neural networks

**absolute eigenvalue sum:**

- define the energy of a graph to be the sum of absolute value of its eigenvalues, \sum_i |\lambda_i|
- for a network with N nodes, the weight matrix is an N x N adjacency matrix with nonnegative real entries (a labeled, weighted graph)
- use this as a regularization term
- begin with small networks and datasets such as MNIST

morally, this is similar to summing the singular values which is just nuclear norm regularization:

https://arxiv.org/abs/2405.14544

**training path energy:**

- let A_0 represent the initial weight matrix, and A_1 represent the trained network
- let A_t be the network at normalized time t during training
- then \gamma: [0,1] --> R^N, \gamma(t) = A_t is a path in weight space
- the energy of a path is defined as (1/2)*\int_[0,1] ||\gamma'(t)||^2 dt
- attempt to minimize this energy by using a local regularier
