---
title: A Comprehensive Guide to Text Classification and Language Modeling in NLP
aliases:
---
# NLP: Text Classification and Language Modelling

Natural Language Processing (NLP) has grown leaps and bounds over the past few years. This guide delves deep into some fundamental concepts like text classification, language modeling, and neural networks, providing a technical yet accessible overview suitable for both beginners and seasoned practitioners.

## Text Classification

### What is Text Classification?

Text classification involves categorizing text into predefined labels based on various traits such as topic, sentiment, subjectivity/objectivity, and intent. For example, determining whether a movie review is positive or negative falls under sentiment analysis, a type of text classification.

### The Basics

At its core, machine learning (ML) for text classification works with paired data $\langle X, Y \rangle$, where $X$ represents the input text, and $Y$ is the corresponding label. The process involves:

1. **Training Data**: Feeding the model with labeled examples.
2. **Learning Algorithm**: Using algorithms to learn patterns from the data.
3. **Feature Extractor**: Identifying relevant features from the text.
4. **Scoring Function**: Assigning scores to different classes based on the features.
5. **Model Preparation**: Building a model that can make predictions on new, unseen data.

Once trained, the model performs inference on the test set to classify new text samples.

### Generative vs. Discriminative Models

#### Generative Models

Generative models calculate the probability of the input data itself. They focus on modeling the joint probability $P(X, Y)$ or the stand-alone probability $P(X)$.

#### Discriminative Models

Discriminative models, on the other hand, calculate the probability of a label given the data, denoted as $P(Y|X)$. They directly model the decision boundary between classes, which often leads to better performance in classification tasks.

### Language Modeling

Language modeling is the task of calculating the probability of a sentence. Formally, it's expressed as:

$$P(X) = \prod_{i=1}^{I}P(x_i|x_1, \ldots, x_{i-1})$$

The challenge lies in predicting these probabilities accurately.

#### Unigram Modeling

A simple approach is unigram modeling, which assumes each word is independent of the others. The probability is estimated using Maximum Likelihood Estimation (MLE):

$$P_{MLE} (x_i) = \frac{c_{train}(x_i)}{\sum_{\tilde{x}}c_{train}(\tilde{x})}$$

However, this method doesn't consider context and assigns zero probability to unseen words in the training data.

#### Handling Unknown Words

To tackle unknown words, models can be character or subword-based, calculating probabilities based on word spelling. Additionally, parameterizing in log space ensures numerical stability by converting multiplication into addition:

$$\log P(X) = \sum \log P(x_i| \text{context})$$

### Generative Text Classifier

A generative classifier decomposes the joint probability as:

$$P(X, y) = P(X|y)P(y)$$

Here, $P(X|y)$ is a class-conditional language model trained on data specific to class $y$, and $P(y)$ represents the prior probability of class $y$:

$$P(y) = \frac{c(y)}{\sum_{\tilde{y}}c(\tilde{y})}$$

#### Naive Bayes Classifier

A common generative classifier is the Naive Bayes classifier, which uses a bag-of-words approach. It multiplies the frequency of each word in the sentence with the log probabilities $log P(x=\text{word}|y)$.

### Discriminative Model Training

Discriminative models train to directly optimize $P(Y|X; \theta)$ with parameters θ\theta. The loss function, often the negative log-likelihood, is minimized:

$\mathcal{L}_{train}(\theta) = - \sum_{\langle X,y \rangle \in \mathcal{D}_{train}} \log P(Y|X; \theta)$


Optimization is typically done using gradient descent:

$$\theta \leftarrow \theta - \alpha \frac{\partial \mathcal{L}_{train}(\theta)}{\partial \theta}$$


#### Bag-of-Words Discriminative Model

For binary classification, the score for a class yy given input XX is:

sy∣X=θy+∑i=1∣X∣θy∣xis_{y|X} = \theta_{y} + \sum_{i=1}^{|X|} \theta_{y|x_i}

This score is converted to a probability using the sigmoid function:

P(y∣X;θ)=11+e−sy∣XP(y|X; \theta) = \frac{1}{1 + e^{-s_{y|X}}}

For multi-class classification, softmax is used instead of sigmoid:

P(y∣X;θ)=esy∣X∑y^esy^∣XP(y|X; \theta) = \frac{e^{s_{y|X}}}{\sum_{\hat{y}} e^{s_{\hat{y}|X}}}

### Evaluation Metrics

- **Accuracy**: The percentage of correctly predicted labels.
$$\text{Accuracy} = \frac{1}{|X|}\sum_{i=1}^{|X|}\delta(y_i = \hat{y_i})$$
- **Precision**: The proportion of true positives among predicted positives.
$$\text{Precision} = \frac{c(y=1,\hat{y}=1)}{c(\hat{y}=1)}$$
- **Recall**: The proportion of true positives among actual positives.
$$\text{Recall} = \frac{c(y=1,\hat{y}=1)}{c(y=1)}$$
- **F1 Score**: The harmonic mean of precision and recall.
$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

## Language Modeling and Neural Networks

### What is a Language Model?

A language model (LM) generates sentences by predicting the next word given the previous context:

$$P(X) = \prod_{i=1}^{I} P(x_i | x_1, \ldots, x_{i-1})$$


### Applications of Language Models

- **Scoring Sentences**: Assessing grammatical correctness.
- **Generating Sentences**: Creating new text by sampling from the probability distribution.

### Smoothing Methods

To handle zero probabilities for unseen words, various smoothing techniques are employed:

- Additive/Dirichlet smoothing
- Discounting
- Kneser-Ney smoothing

### Types of Language Models

- **Class-Based Language Models**
- **Skip-Gram Models**

### When to Use N-gram Models?

N-gram models are extremely fast and effective for modeling low-frequency phenomena, as count-based models never forget seen data.

### Evaluation of Language Models

- **Log-Likelihood**:
$$\mathcal{LL}(\mathcal{E}_{test}) = \sum_{E \in \mathcal{E}_{test}} \log P(E)$$
- **Per-Word Log Likelihood**:
$$\mathcal{WLL}(\mathcal{E}_{test}) = \frac{1}{\sum_{E \in \mathcal{E}_{test}} |E|} \sum_{E \in \mathcal{E}_{test}} \log P(E)$$
- **Perplexity**:
$$\text{ppl}(\mathcal{E}_{test}) = e^{-\mathcal{WLL}(\mathcal{E}_{test})}$$

### Handling Unknown Words

Unknown words are inevitable. Common strategies include:

- Limiting vocabulary by frequency threshold.
- Modeling characters or subwords to capture the probability of unseen words.

## Neural Networks in NLP

Neural networks are essentially computation graphs, typically directed and acyclic. Smaller graphs are easier to compute, which is crucial for efficiency.

### Sequence Modeling and Recurrent Networks

Language inherently involves long-distance dependencies like agreement in number and gender or selectional preferences. Recurrent Neural Networks (RNNs) and their variants like LSTMs are designed to handle such dependencies.

#### Bidirectional RNNs

These process the sequence in both forward and backward directions, allowing the model to capture context from both sides.

#### Long Short-Term Memory (LSTM)

LSTMs mitigate the vanishing gradient problem, enabling the model to learn long-term dependencies effectively.

### Increasing Efficiency

- **Mini-Batching**: Processing multiple samples simultaneously.
- **Masking**: Padding sequences to uniform lengths and masking the padding tokens.
- **Truncated Backpropagation Through Time (BPTT)**: Handling long sequences by truncating the backpropagation process.

## Conditioned Generation

### Generating Sentences

Generating sentences involves sampling or selecting words based on the probability distribution:

- **Ancestral Sampling**: Sampling words one by one until an end-of-sentence token is generated.
- **Greedy Search**: Selecting the highest probability word at each step.
```
while yj−1≠"</s>":
	yj=arg⁡max ⁡P(yj∣X,y1,…,yj−1)
```
- **Beam Search**: Keeping multiple hypotheses at each step to explore more possibilities.

### Examples of Conditioned Generation

- **Translation**
- **Summarization**
    - Extractive: Copying input.
    - Abstractive: Generating new text.
- **Dialog Response Generation**
- **Image Captioning**

### Evaluation Metrics

#### BLEU (Bilingual Evaluation Understudy)
Measures n-gram overlap with reference translations.
**Pros**:
- Easy to use.
- Good for measuring system improvements.
**Cons**:
-  Doesn't always match human judgment.
- Penalizes correct paraphrases.
#### **Embedding-Based Metrics**:
- **BERTScore**: Uses BERT embeddings to find similarity.
- **BLEURT**: Trains BERT to predict human evaluation scores.
- **COMET**: Combines source sentences with reference outputs.
- **PRISM**: Based on a paraphrasing model.
- **BARTScore**: Calculates the probability of source, reference, or system output.

### Challenges with Beam Search

- **Machine Translation**: Tends to produce short hypotheses.
- **Open-Ended Generation**: Can lead to repetition.

Other sampling methods like top-k and nucleus sampling are often used to mitigate these issues.

## Attention Mechanisms

### What is Attention?

Attention allows models to focus on different parts of the input when generating each part of the output. Instead of compressing the entire input into a single vector, attention uses multiple vectors proportional to the input length.

### Calculating Attention

1. **Query and Key Vectors**: Each word is encoded into vectors.
2. **Attention Weights**: Calculated using a score function a(q,k)a(q, k).
3. **Softmax Normalization**: Ensures weights sum to one.
4. **Weighted Sum**: Combines value vectors based on attention weights.

### Attention Score Functions

- **Multilayer Perceptron**:
$$ a(q, k) = w_2^T \tanh(W_1 [q; k]) $$
- **Bilinear**:
$$ a(q, k) = q^T W k $$
- **Dot Product**:
$$ a(q, k) = q^T k $$
- **Scaled Dot Product**:
$$ a(q, k) = \frac{q^T k}{\sqrt{|k|}} $$

### Self-Attention and Multi-Headed Attention

- **Self-Attention**: Each word attends to all other words in the sentence, creating context-sensitive encodings.
- **Multi-Headed Attention**: Multiple attention "heads" focus on different parts of the sentence independently, enhancing the model's ability to capture diverse relationships.

## Transformers

### The Transformer Model

Transformers are sequence-to-sequence models based entirely on attention mechanisms, making them highly parallelizable and efficient. The main components include:

1. **Multi-Headed Attention**
2. **Feed Forward Networks**
3. **Add & Normalize Layers**

### Training Transformers

- **Positional Encoding**: Since transformers don't have a built-in sense of order like RNNs, positional encoding helps distinguish word positions.
- **Layer Normalization**: Ensures that activations remain in a reasonable range, stabilizing training.
- **Specialized Training Schedules**: Adjust learning rates and other hyperparameters for optimal performance.
- **Label Smoothing**: Introduces uncertainty, preventing the model from becoming overconfident.
- **Masking**: Ensures that the model doesn't attend to future tokens during training.

## Language Model Pretraining

### Pre-training Methods

Pre-training involves training models on large datasets before fine-tuning them on specific tasks. Common approaches include:

- **Standard Multi-Task Learning**: Training on multiple tasks simultaneously.
- **Pre-train and Fine-Tune**: Pre-training on a general task, then fine-tuning on a specific one.

### Popular Pre-training Objectives

- **Auto-Regressive Language Modeling**: Suitable for text generation tasks.
- **Masked Language Modeling (MLM)**: Used in models like BERT for tasks requiring understanding of context.

### Notable Pre-trained Models

- **BERT**: Uses MLM and next sentence prediction.
- **RoBERTa**: Optimizes hyperparameters and drops next sentence prediction for better performance.
- **XLNet**: Extends BERT by predicting words in random order, capturing longer contexts.
- **DeBERTa**: Introduces disentangled attention mechanisms.
- **ALBERT**: Reduces model size with parameter sharing.
- **DistilBERT**: A smaller, faster version of BERT trained to match its performance.
- **GPT Series**:
    - **GPT-2**: 1.5B parameters, focuses on left-to-right language modeling.
    - **GPT-3**: 175B parameters, excels in text generation.
- **PaLM**: A massive 540B parameter model designed for diverse language tasks.

## Conclusion

This guide has covered essential aspects of text classification and language modeling within NLP, from foundational concepts to advanced neural network architectures like transformers. Understanding these principles is crucial for developing robust NLP applications that can interpret and generate human language effectively.

# Further Reading

- [Attention is All You Need](https://arxiv.org/abs/1706.03762) by Vaswani et al.

# References

- [CMU Advanced NLP 2022](https://youtube.com/playlist?list=PL8PYTP1V4I8D0UkqW2fEhgLrnlDW9QK7z&si=UlAoO4C0aIrDQzXt)

# Tags

#NLP #ArtificialIntelligence #Transformers
