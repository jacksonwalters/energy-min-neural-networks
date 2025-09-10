# regularization-neural-networks

Uses various regularization techniques to minimize "energy" in neural networks.

---

**spectral regularization:**

- define the energy of a graph to be the sum of absolute value of its eigenvalues, \sum_i |\lambda_i|
- for a network with N nodes, the weight matrix is an N x N adjacency matrix with nonnegative real entries (a labeled, weighted graph)
- use this as a regularization term
- begin with small networks and datasets such as MNIST

morally, this is similar to summing the singular values which is just nuclear norm regularization.

**references:**

- Nuclear Norm Regularization for Deep Learning - Christopher Scarvelis, Justin Solomon (NeurIPS 2024)
    - The Jacobian nuclear norm encourages functions to vary only along a few directions, but directly computing it is intractable in high dimensions. This work introduces efficient approximations—using Frobenius norms of component Jacobians and a denoising-style estimator—that make Jacobian nuclear norm regularization practical and scalable for deep learning.
    - https://arxiv.org/abs/2405.14544

---

**training path energy:**

- let A_0 represent the initial weight matrix, and A_1 represent the trained network
- let A_t be the network at normalized time t during training
- then \gamma: [0,1] --> R^N, \gamma(t) = A_t is a path in weight space
- the energy of a path is defined as (1/2)*\int_[0,1] ||\gamma'(t)||^2 dt
- attempt to minimize this energy by using a local regularization term

**references:**

- Path-SGD: Path-Normalized Optimization in Deep Neural Networks — Neyshabur, Salakhutdinov & Srebro (NeurIPS 2015).
    - Introduces path-style norms and optimization geometry that are invariant to layer rescaling; shows how a path-based regularizer leads to practical optimization rules. 
    - https://arxiv.org/abs/1506.02617
- Understanding Natural Gradient in Sobolev Spaces — Bai, Rosenberg & Xu (2022, arXiv).
    - Derives Sobolev-type (i.e. derivative-penalizing) natural gradients: directly relevant when you want to measure path length / energy in a Sobolev metric and to change geometry of updates. (Note: Steven Rosenberg is a coauthor.) 
    - https://arxiv.org/abs/2202.06232
- Neural Network Training Techniques Regularize Optimization Trajectory: An Empirical Study — Chen et al., arXiv 2020.
    - Empirically studies how common tricks (BN, momentum, etc.) impose regularity on optimization trajectories; formalizes a trajectory-regularity principle and links it to convergence. Good for motivation / empirical phenomena.
    - https://arxiv.org/abs/2011.06702
- Penalizing Gradient Norm for Improving Generalization — Zhao et al. (2022).
    - Instead of penalizing parameter motion directly, this paper penalizes the gradient norm (a different but related kind of trajectory regularization) and shows improved generalization. Useful as an alternative objective that’s easier to implement. 
    - https://arxiv.org/abs/2202.03599
- Regularizing Trajectories to Mitigate Catastrophic Forgetting / Co-natural gradient — Michel et al. (approx. 2020–2021).
    - Papers in this vein regularize optimization trajectories directly (for continual learning) — useful references for how people add trajectory penalties in practice.
    - https://pmichel31415.github.io/assets/conatural_gradient.pdf?utm_source=chatgpt.com
