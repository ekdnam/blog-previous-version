---
title: "AI Efficiency II: Pruning and Sparsity"
---
# Introduction

Our AI models are getting very big these days, going back to the scaling hypothesis. As models get bigger their accuracy on tasks keeps increasing, which is a good thing. However, the bigger the model, it also leads to a larger memory footprint to store a model on disk. However, the rate of increase in GPU memory has not kept up with the scale up in number of parameters. Also, memory movement is pretty power hungry.<sup>1</sup> Leading to how power consumption if we have to continuously move parameters from HBM to SRAM and vice-a-versa. 

![[../images/efficiency/2_pruning/1_memory_power_consumption.png]]

<sub>(Image from 2)</sub>

Thus, we would like to explore methods by which we can reduce either the number of parameters, or store the parameters in a different format, both while trying to maintain the accuracy. The first is called pruning while the latter is called quantization. In this blog we will be covering pruning.

# What is neural network pruning?

We can formulate pruning as 
$$\mathop{\arg\min}_{W_p} \space L(X; W_p) $$
subject to 
$$\| \mathbf{W_p} \|_{0} < N$$
- $L$ is the objective function for neural network training
- $\mathbf{x}$ is input, $\mathbf{W}$ is the original weights, $\mathbf{W_p}$ is the pruned weights.
- $\| \mathbf{W_p} \|_{0} \space$ is the number of nonzero parameters in the network, $N$ is our target number of nonzero parameters.

Thus we want to have the minimum loss with our pruned weights (non zero weights) with the number of nonzero params being below a certain threshold. Thus we can observe we want to make the network more **sparse** or just remove weights.

# How to prune?

We can prune a neural network either by pruning neurons, or pruning synapses.

**Neuron pruning** means removing entire neurons if they don't contribute much to the network's function. This could mean fewer layers (depth pruning) or fewer neurons per layer (width pruning), making the network simpler and faster. The idea is that some neurons might be redundant, so removing them keeps performance while reducing complexity.

**Synapse pruning** involves cutting connections between neurons with very low weights, assuming they don't significantly affect the output. This makes the network sparser, potentially improving speed and reducing overfitting by making the model less complex.
Both methods aim for efficiency, but they affect how the network processes information differently. Neuron pruning might change the network's structure more dramatically, while synapse pruning tweaks the existing connections.
![[../images/efficiency/2_pruning/2_pruning_neurons_synapses.png]]

<sub>(Image from 4)</sub>

One of the reasons we don't want to keep on trivially keep on removing neurons is because of its impact on accuracy. _4_ talks more about this. 

![[../images/efficiency/2_pruning/3_prune_and_finetune.png]]

<sub>(Image from 2)</sub>

_4_ first trains a baseline network (AlexNet). 

**Pruning**: They then gradually prune the network, by removing weights whose absolute value is near to zero, under a threshold. As we see, the more we prune, the more the loss in accuracy (pruning). 

**Pruning + Finetuning**: If we train the remaining weights that survive the pruning, we can prune more, leading to higher accuracy given same pruning ratio. The weight distribution becomes smoother as well. We can prune away ~80% of the weights without hurting the accuracy.

**Iterative Pruning and Finetuning**: We can do an another round of pruning and finetuning, doing it iteratively, and get to about ~90% pruning without hurting the accuracy.

# Pruning in the industry

NVIDIA<sup>5</sup> has introduced structured sparsity in its hardware, starting from the Ampere architecture. With the third generation Tensor Cores in A100s, we can take advantage of sparsity to 2x the max throughput while maintaining the accuracy of the MACs (Matrix Multiply Accumulate).

![[../images/efficiency/2_pruning/4_nvidia_sparsity_diagram.jpg]]

<sub>(Image from 6)</sub>


## Structured sparsity

![[../images/efficiency/2_pruning/5_structured_sparsity_pattern.png]]

<sub>(Image from 5)</sub>

NVIDIA Ampere and Hopper architectures GPUs add fine-grained structured sparsity, taking advantage of sparse Tensor Cores, which require a 2:4 sparsity pattern. This means that among a group of 4 contiguous values, atleast 2 must be zero, leading to a 50% sparsity rate. We can extend this to N:M sparsity, where in a group of M contiguous values, N should be zero. After compression, only non-zero values and the associated metadata gets stored. This can be primarily applied to fully connected layers or convolutional layers. 

Sparse Tensor Cores accelerate this format by operating only on the nonzero values in the compressed matrix. They use the metadata that is stored with the nonzeros to pull only the necessary values from the other, uncompressed operand. So, for a sparsity of 2x, they can complete the same effective calculation in half the time.<sup>7</sup>
 ![[../images/efficiency/2_pruning/7_dense_sparse_matrix_nvidia.png]]

<sub>(Image from 8)</sub>

Lets consider a GEMM (Generalized Matrix Multiply),where we multiply sparse $A \in \mathbb{R}^{M \times K}$ (which has 2:4 sparsity) with a dense $B \in \mathbb{R}^{K \times N}$. This works on the sparse tensor core by converting the GEMM  from dense to sparse. $A$ is converted from $M \space \text{x} \space K$ to $M \space \text{x} \space \frac{K}{2}$ after since its 50% sparse. The tensor core selects only the values from $B$ that correspond to the nonzero values in $A$, skipping unnecessary multiplications by 0.<sup>8</sup>

Sparsity motivates engineers and scientists to prune their networks to make use of this speedup.


# Pruning at different granularities

![[../images/efficiency/2_pruning/6_pruning_granularities.png]]

<sub>(Image from 2)</sub>

We consider the case of convoulutional neural networks, which have kernels. Kernels in CNNs extract features from input images by applying convolution operations, detecting patterns like edges or textures.

## Fine-grained pruning
- We select "redundant" neurons based on some heuristic (discussed later).
- Since its flexible, leads to higher compression ratio.
- However, flexibility means we can't make use of structured sparsity on GPUs trivially.
## Pattern-based pruning
- This is essentially N:M sparsity, discussed [before](#structured-sparsity).
- 2:4 works great on NVIDIA GPUs.
## Vector-based pruning
- We make entire rows sparse in a kernel ie we remove neurons.
## Kernel-based pruning
- We make entire kernels sparse.
## Channel-based pruning
- Specifically targets the channels of convolutional layers in CNNs. 
- Each channel in a convolutional layer corresponds to a feature map, and pruning involves removing entire channels rather than individual weights or kernels.

# Pruning Criterion

If we remove the less important neurons and synapses, the accuracy of the neural network is higher. How to decide which neuron is less important?
<sub>(All the images in this section are from 2)</sub>

## Magnitude-based pruning

We consider weights with higher absolute values to be more important. 

### Element-wise pruning

We zero out elements of the matrix whose value is below a certain threshold.

![[../images/efficiency/2_pruning/8_element_pruning.png]]
### Row-based pruning

We can use three heuristics: $l1$, $l2$, or the generalized $l_p$ norm

#### L1-norm

We calculate $\text{Importance} = \sum_{i \in S} | w_i |$ where $\mathbf{W}^{s}$ is the structural set of parameters in $\mathbf{W}$

![[../images/efficiency/2_pruning/9_row_based_pruning_l1.png]]

#### L2-norm

$\text{Importance} = \sqrt {\sum_{i \in S}  w_{i}^2 }$ 
![[../images/efficiency/2_pruning/10_row_based_pruning_l2.png]]

#### $L_p$-norm

$\text{Importance} = ({\sum_{i \in S}  (| w_{i} |)^p)^{\frac{1}{p}} }$ 



# Acknowledgements
This series of blogs on AI efficiency is heavily inspired by [Dr. Song Han's](https://hanlab.mit.edu/songhan) course on [TinyML and Efficient Deep Learning Computing](https://hanlab.mit.edu/courses/2024-fall-65940). For the papers we discussed, I referred [horseee/Awesome-Efficient-LLM](https://github.com/horseee/Awesome-Efficient-LLM) on Github.

I used the help from my buddies (sorry for anthromorphizing) [Grok-2](https://x.ai/blog/grok-1212), [OpenAI O1](https://openai.com/o1/), [gemini-2.0-flash-thinking-exp-01-21](https://deepmind.google/technologies/gemini/flash-thinking/) for this post. 

# References
1. [Computing’s Energy Problem (and what we can do about it)](https://gwern.net/doc/cs/hardware/2014-horowitz-2.pdf)
2. [EfficientML.ai Lecture 3 - Pruning and Sparsity Part I (MIT 6.5940, Fall 2024)](https://www.youtube.com/watch?v=EjsB0WgIfUM)
3. [horseee/Awesome-Efficient-LLM](https://github.com/horseee/Awesome-Efficient-LLM)
4. [Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626)
5. [Structured Sparsity in the NVIDIA Ampere Architecture and Applications in Search Engines](https://developer.nvidia.com/blog/structured-sparsity-in-the-nvidia-ampere-architecture-and-applications-in-search-engines/)
6. [How Sparsity Adds Umph to AI Inference](https://blogs.nvidia.com/blog/sparsity-ai-inference/)
7. [Accelerating Inference with Sparsity Using the NVIDIA Ampere Architecture and NVIDIA TensorRT](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/)
8. [Accelerating Sparse Deep Neural Networks](https://arxiv.org/abs/2104.08378)


# Tags
#efficiency #ai #gpu #nvidia