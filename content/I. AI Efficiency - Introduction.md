# Scaling Hypothesis

In the era of LLMs, we have seen the [scaling hypothesis](https://gwern.net/scaling-hypothesis) hold true. Very simply, it states that as we train larger and larger neural networks, more and more sophisticated behaviour emerges to be the easiest way to optimize for all different types of tasks & data. This is quite similar to the human brain, which looks very much like a scaled up version of a primate brain. This was seen at first stated in the paper "[Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361)" by OpenAI circa 2020. They said that "**the loss scales as a power-law with model size, data size, and amount of compute used for traning...**". 
Where loss is cross-entropy loss for the test set. (LOOK INTO THIS)

The entire LLM arms race across the world, NVIDIA's crazy valuation, Intel's bad days, rising need for energy (with Google, Microsoft, Amazon planning to recommision nuclear reactors), geopolitical ramifications (why China cannot get NVIDIA GPUs beyond a certain limit), why LLAMA models are released in 7B 14B 70B and 405B and much more is a downstream effect  of this statement. 

I will go into why I am saying so in other blogs of this series.

![[scaling-hypothesis.png]]

As we can see, as the dataset size (D) and parameters of the model (N, also called model size) increases, which means we also have to increase the compute (C). We scale them up. Then the test loss decreases ie we can say the models get better. We can also usethis build a good enough estimate of what performance will a model have a specific configuration of N, D and C.

Thus, researchers used this to estimate how much N and D you need given C (since GPUs are limited) - assuming you have infinite data, you can effectively build models of certain size. As N increases, performance increases. Thus we see LLama 405B performs better on tasks than LLama 70B, which performs better than 13B and so on. 

# Model precision
A 405B parameter model is an insane amount of parameters. These parameters are trained when we are training the model, and when we have to save it to disk, we have a couple of options.

## Floating point precision
We have different ways in which we can store a model.
1. FP32 - uses 32 bits to represent a number (highest precision) - 4 bytes
2. FP16 - uses 16 bits - 2 bytes
3. FP8 - uses 8 bits - 1 byte
Since 1 byte is 8 bits.

The more number of bits used to store a number, the higher the accuracy of the representation. However, it also means more memory is required. And at billion parameter scale, it starts to make a difference. 

If we are saving the parameters in FP32 (floating point 32) format - a higher form of precision - which means an object in FP32 takes up 32 bits of memory (people don't store models in FP32 these days, we will get into it in a later blog, but this is just for an example), then the size of the model will be $405 (N) * 32 (dtype \space size) / 8 (number \space of \space bits \space per \space byte) = 405 * 4 = 1620 bytes = ~ 1.63 TB$
If we wanted to store and load this model from disk say HDD, then 1.6 GB is not a big number. But, it is not so simple.

# AI accelerators
When we are training an AI model, specifically a neural network, we need to perform a forward pass and a backward pass through the model. In a forward pass, we pass our input (might be text, images, audio, video etc) through the different layers and components to compute an "output", as well as an output loss (there are different evaluation metrics like binary cross entropy, categorical cross entropy, Kullback-Leibler Divergence loss, and many more). The use of this loss is to compute the difference between our prediction and the real value. The lesser the loss the better. Then in the backward pass, we backpropagate this loss to update our weights (parameters) such as to reduce the loss using some learning rate and an optimizer. We do this using the chain rule. 

In the serving phase, we only perform the forward pass. 

The forward and backward pass are very computationally intensive - requiring trillions of operations. For example, the GPT 3 (175B) training took 3.14E23 FLOPS of computation for training. (FLOPS means floating point operations - like addition, multiplication, exponentiation)

Also, since the GPT 3 model size is 175B, it will take 175 * 4 = 700 GB if the model is stored in FP32.

If we want to train such a model fast (efficiently), we face 3 main bottlenecks.
1. Compute: How fast can we perform the forward and backward pass?
2. Memory (bandwidth): How fast can we load the model from disk on to the hardware accelerator?
3. Memory (size): How large is the memory of the hardware accelerator? (Although this not something which is usually talked about in efficiency, I will discuss to build intuition)

> We want to train the models fast so that - 1) we can iterate faster to lead to better results 2) be first to release a model in the LLM market (this matters a lot to capture market share re: ChatGPT)

Here come in AI accelerators.

These are pieces of hardware who are hyperoptimized for AI workloads, so that we can train and serve a model fast.

## Introduction to GPUs

Graphics Processing Units (GPUs) are specialized hardware accelerators designed to perform parallel processing tasks efficiently. Originally built to handle graphics rendering, GPUs have evolved into powerful tools for general-purpose computing, particularly for tasks requiring massive parallelism, such as machine learning, scientific simulations, and cryptography. Understanding the key components of GPU architecture, including memory systems and specialized processing cores, is essential to leveraging their full potential.

---

### High Bandwidth Memory (HBM)

One of the most critical components of a GPU's architecture is its memory system. **High Bandwidth Memory (HBM)** is an advanced type of GPU memory that significantly boosts memory bandwidth compared to traditional GDDR memory. Unlike GDDR, which places memory chips around the GPU die, HBM stacks memory vertically on the same package as the GPU. This stacked design allows for **wider memory interfaces** (thousands of bits) and shorter distances between the GPU and memory, resulting in **higher bandwidth and lower power consumption**.

HBM is crucial for applications that need to process large amounts of data quickly, such as AI model training, where the GPU must continuously access and manipulate massive datasets. By providing **terabytes-per-second memory bandwidth**, HBM ensures that the GPU cores have fast access to the data they need, reducing latency and improving overall performance.

---

### Tensor Cores

Modern GPUs, especially those designed for AI and machine learning workloads, include **Tensor Cores**, a specialized type of processing unit designed for efficient matrix operations. Tensor Cores are optimized for **tensor computations**, which are fundamental to deep learning models. These cores handle **mixed-precision arithmetic**, using lower precision (such as FP16) for faster computation while maintaining accuracy through higher-precision accumulation (such as FP32).

The introduction of Tensor Cores has dramatically accelerated AI workloads by allowing GPUs to perform matrix multiplications and accumulations in a single operation. For tasks like training neural networks or performing inference, Tensor Cores can deliver significant performance improvements compared to traditional CUDA cores.

---

### GPU Memory Bandwidth and Its Importance

While Tensor Cores can perform a massive number of computations in parallel, their performance is only as good as the **rate at which data can be supplied to them**. This is where **GPU memory bandwidth** comes into play. GPU memory bandwidth refers to the rate at which data can be transferred between the GPU’s memory (VRAM) and its processing cores, typically measured in **gigabytes per second (GB/s)**.

For Tensor Cores to operate at full capacity, they require a steady stream of data. If the memory bandwidth is too low, the cores may sit idle, leading to underutilization and reduced performance. **HBM addresses this challenge** by providing ultra-high bandwidth that matches the data demands of Tensor Cores, ensuring the GPU can handle data-intensive tasks efficiently without bottlenecks.

## NVIDIA H200

I will give an example using the NVIDIA H200. Lets take a look at its [datasheet](https://nvdam.widen.net/s/nb5zzzsjdf/hpc-datasheet-sc23-h200-datasheet-3002446). 

![[H200-datasheet.png]]

## Compute

On the FP32 cores, at maximum efficiency (which is hard to achieve) we can perform 989 TFLOPS ie 989E12 floating point operations per second. This means if we have only a single H200 (not considering memory requirements), we'll need around ~10 years to train the model. 

Code from ChatGPT on how to compute this
```python
total_flops = 3.14e23  # Total FLOPs required for training GPT-3
hardware_tflops = 989  # Hardware performance in TFLOPs

hardware_flops = hardware_tflops * 1e12 # Convert hardware performance to FLOPs (1 TFLOP = 1e12 FLOPs)

training_time_seconds = total_flops / hardware_flops # Calculate training time in seconds

training_time_days = training_time_seconds / (60 * 60 * 24) # Convert time to more readable units

training_time_days # (~3674.68 days)

```
We cannot wait ten years to train a model! 

Now lets consider FP8 tensor cores which have 3958 TFLOPS. 

We can simply divide the FP32 time by 989 and multiply by 3958, which will result in approx ~2.5 years. Although this is still not feasible, its a much better improvement!

## Memory (size)
The H200 memory is 141GB. A GPT 3 which takes 700GB in FP32 cannot be stored entirely on a single GPU. Thus we need multiple GPUs to store a large model in one go, in this case 5. However, if the GPU memory was larger, we will lead lesser number of GPUs.

## Memory (bandwidth)
As we can see, the memory bandwidth is 4.8TB/s. Lets consider the case where we are using only a single H200. 

## Time to train
Lets consider we can trivially shard a model into equally or nearly equal chunks of memory and compute. Lets consider we have sharded a model into n chunks.

Time to train on a single GPU = n * (time to compute + time to load model from DDR * 2)
The 2 comes from time to load and unload a specific chunk, so that we can load the next one.

Thus, time is directly proportional to n, compute, and memory bandwidth. These are bottlenecks.

To counteract n, you can simply get more GPUs (datacenters) and load a model across multiple GPUs - thus having more memory across - leading to different types of parallelisms (in this case model parallelism) to work with these model chunks in one go, and also, communication collectives, to communicate among these GPUs.

To counteract compute, you need efficient architectures so that you can have lower FLOPS, and use lower precision since GPUs have higher FLOPS at lower precision.

To counteract memory, you need GPUs with more bandwidth out of the box, and also reduce the number of times to load the weights from DDR.