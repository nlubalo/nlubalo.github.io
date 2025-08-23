---
layout: post-product-queen
title:  "Attention Mechanism"
image: assets/images/home.jpg
description: "After mastering tokenization and embeddings, the next step in understanding Transformers is the attention mechanism. Attention allows models to focus on relevant parts of the input when making predictions, enabling them to capture context and relationships between words effectively."
categories: [projects]
---

<p>
In the first section, we built a foundation of tokenization and creating embeddings where sentences were broken down into tokens and each token was represented as a vector in high-dimensional space. In attention, for each token, the model creates threes vectors called:
<ul>
<li>
    Query (Q)
    </li>
    <li>
    Key (K)
    </li>
    <li>
     Value (V)
    </li>
</ul>
<p>
Each token has these three vectors associated with it. The attention mechanism uses these vectors to determine how much focus to place on other parts of the input sequence when processing a particular token. This is crucial for understanding context and relationships in language.
</p>
</p>

### Attention via the Social Graph Analogy
Think of yourself as the Query (Q).
Everyone you meet is a Key (K) carrying their own traits and roles.
 - When you interact, you are comparing your current needs or context (Q) against what each person (K) represents.
 - Based on this similarity, you decide how much attention (weight) to give them.
 - The actual knowledge or help you get in them is the Value (V)

Just like in real life:
 - You might be a manager at work - > You give more attention to certain colleagues (work context)
 - But the same colleague might be your gym instructor -> In the context, you attend/relate to then differently.

Thats exactly what Transformers do with words in a sentence.
 - The Q-K interaction build a network of context-depenedent connections
 - The attention score change depending on the sentence or task
 - The  weighted sums of Vs gives the model flexible, context-aware memory

-----
✨ In short: Attention in Transformers is like constantly rebuilding a social graph where connections (Q–K) depend on context. The model learns these shifting relationships and uses them to decide which information (V) matters most at each step.

-----

#### Visualizing attention as a social graph
![Attention as Social Graph](/assets/images/attention_social graph.png)

- You (the Query) are trying to decide who to “pay attention” to.

- Each person (the Keys) pulls your attention with different strengths depending on the context.

- The arrow thickness shows how much attention you give: stronger to the instructor, weaker to your friend.

This mirrors how Transformers weigh words differently when processing a sentence.

### Having looked at the social graph analogy, let's looks at the math behind attention.

![Attention as Social Graph](/assets/images/attention_image.png)
From left to right, the image above shows the full artchitecture of a Transformer block, zooming into the attention mechanism for multihead attention and into scaled dot-product attention.

We will start with scaled dot-product attention, the core of the attention mechanism.

The core idea is to compute attention scores between the Query and all Keys using the dot product, scale them, apply a softmax to get weights, and then use these weights to compute a weighted sum of the Values.
Here’s the step-by-step breakdown:

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$


where:

1. **Dot Product**: Compute the dot product between the Query (Q) and each Key (K). This gives a measure of similarity or relevance between the Query and each Key.
$ \text{scores} = QK^T $
2. **Scaling**: Scale the scores by the square root of the dimension of the Key vectors (d_k). This helps stabilize gradients during training.
   $\
\text{scaled\_scores} = \frac{\text{scores}}{\sqrt{d_k}}
\$

3. **Softmax**: Apply the softmax function to the scaled scores to get attention weights. This converts the scores into probabilities that sum to 1.
   $\\text{weights} = \text{softmax}(\text{scaled\_scores})\$
4. **Weighted Sum**: Use the attention weights to compute a weighted sum of the Value (V) vectors. This gives the final output of the attention mechanism.
   $
   \\text{output} = \text{weights} \cdot V\
   $

### Implementing Scaled Dot-Product Attention in Python
Here’s a simple implementation of scaled dot-product attention in Python using NumPy:
```python
import numpy as np
def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]  # Dimension of the Key vectors

    # Step 1: Compute the dot products between Q and K
    scores = np.matmul(Q, K.transpose(-2, -1))  # Shape: (num_queries, num_keys)

    # Step 2: Scale the scores
    scaled_scores = scores / np.sqrt(d_k)

    # Step 3: Apply softmax to get attention weights
    weights = np.exp(scaled_scores - np.max(scaled_scores, axis=-1, keepdims=True))
    weights /= np.sum(weights, axis=-1, keepdims=True)  # Shape: (num_queries, num_keys)

    # Step 4: Compute the weighted sum of V
    output = np.matmul(weights, V)  # Shape: (num_queries, value_dim)

    return output, weights
```

### How are these Q, K, V vectors created / trained?
In practice Q, K, V vectors are initilized randmoly and learned during training via backpropagation. The model adjusts the weights used to generate these vectors to minimize the loss function for the specific task (e.g., language modeling, translation).


In the next post, we will look at how to extend this to multi-head attention and build a full Transformer block.