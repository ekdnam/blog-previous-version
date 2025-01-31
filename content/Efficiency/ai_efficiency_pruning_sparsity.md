---
title: "AI Efficiency II: Pruning and Sparsity"
---
# Introduction

Our AI models are getting very big these days, going back to the scaling hypothesis. As models get bigger their accuracy on tasks keeps increasing, which is a good thing. However, the bigger the model, it also leads to a larger memory footprint to store a model on disk. However, the rate of increase in GPU memory has not kept up with the scale up in number of parameters. Also, memory movement is pretty power hungry.<sup>1</sup> Leading to how power consumption if we have to continuously move parameters from HBM to SRAM and vice-a-versa. 

![[./images/efficiency/2_pruning/1_memory_power_consumption.png]]

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
# Acknowledgements
This series of blogs on AI efficiency is heavily inspired by [Dr. Song Han's](https://hanlab.mit.edu/songhan) course on [TinyML and Efficient Deep Learning Computing](https://hanlab.mit.edu/courses/2024-fall-65940). For the papers we discussed, I referred [horseee/Awesome-Efficient-LLM](https://github.com/horseee/Awesome-Efficient-LLM) on Github.

I used the help from my buddies (sorry for anthromorpizing) [Grok-2](https://x.ai/blog/grok-1212), [OpenAI O1](https://openai.com/o1/), [gemini-2.0-flash-thinking-exp-01-21](https://deepmind.google/technologies/gemini/flash-thinking/) for this post. 

# References
1. https://gwern.net/doc/cs/hardware/2014-horowitz-2.pdf
2. https://www.youtube.com/watch?v=EjsB0WgIfUM
3. https://github.com/horseee/Awesome-Efficient-LLM
4. 


# Tags
#efficiency #ai 