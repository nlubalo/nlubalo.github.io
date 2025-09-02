---
layout: post-product-queen
title:  "Multi-Head Attention"
image: assets/images/home.jpg
description: "In scaled dot product attention, we compute attention using as single set of queries, keys and values. But language and context are multi-faceted. Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions by using multiple sets of Q, K, V matrices that are learned during training."
categories: [projects]
---

### Recap: Single Head Attention
In the previous post, we discussed the attention mechanism where for each token, we create three vectors: Query (Q), Key (K), and Value (V). The attention scores are computed using the dot product of Q and K, scaled by the square root of the dimension of K, followed by a softmax to get weights, which are then used to compute a weighted sum of the Vectors.

### Why Multi-Head Attention?
Single-head attention can be limiting because it forces the model to focus on a single representation of the input at a time. However, language is complex and multi-faceted. Different words and phrases can have multiple meanings depending on context. Multi-head attention allows the model to attend to different parts of the input simultaneously, capturing various aspects of the data.

### Intuitive Example
Sentence: "The cat sat on the mat."
    - Head 1 might focus on subject → verb relationships.
    - Head 2 might focus on word position (e.g., "the" often appears at the start).
    - Head 3 might focus on semantic similarity ("cat" ↔ "mat").

By combining them, the model builds a richer representation of the sentence.

Specifically how the multi-head attention works:
1. **Multiple Sets of Q, K, V**: Instead of having a single set of Q, K, and V matrices, multi-head attention uses multiple sets (or "heads") of sub-queries, sub-keys, and sub-values. Each head has its own learned linear transformations for Q, K, and V.
2. **Parallel Attention**: Each head is passed through the scaled dot product attention and computes attention independently, allowing the model to focus on different parts of the input sequence.
3. **Concatenation and Linear Transformation**: The outputs from all heads are concatenated and passed through a final linear layer to produce the final output.

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$

where each head is:

$$
\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)
$$

and

$$
\text{Attention}(Q, K, V) = \text{softmax}\Big(\frac{Q K^\top}{\sqrt{d_k}}\Big) V
$$

Expressed in a computational graph, it is visualized as below (figure credit - Vaswani et al., 2017).

<div class="image-center">
  <img src="{{ '/assets/images/multihead_attention.svg' | relative_url }}" alt="Multihead attention" class="img-fluid">
</div>

### How Multi-Head Attention Works When We Don’t Have Explicit Queries, Keys, and Values

One common question about the Transfomer architecture is how do we get the Q, K, V vectors when we don’t have explicit queries, keys, and values as inputs?

The clever trick is that we don't actuall neead exteranl queries, keys or values. Instead, we reuse the same feature map from the nueral network and tet the model learn how to separate them into Q, K, and V.


### Implementing Multi-Head Attention in Python