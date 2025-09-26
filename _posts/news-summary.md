---
layout: post-product-queen
title:  "Fine-Tuning DistilBART for News Summarization: Lessons From My First Attempt"
image: assets/images/home.jpg
description: "I’ve been experimenting with fine-tuning DistilBART on a custom dataset of news articles. The goal? Teach the model to generate short, meaningful summaries of news content. This post documents what I did, the results I got, and why they weren’t as good as I hoped and my next steps."
categories: [projects]
---

Over the past few weeks, I’ve been experimenting with fine-tuning DistilBART on a custom dataset of news articles I collected and preprocessed. The goal? Teach the model to generate short, meaningful summaries of news content.

This post documents what I did, the results I got, and why they weren’t as good as I hoped and my next steps to improve the process.

---

### 1. The Setup

I started with a news aggregator pipeline:

 - Pull articles via the News API
 - News API truncates very long contents so I used trafilatura to scrap the contents
 - Used article content as input and title as the summary. Concatenated title + content
 - Preprocess by removing white spaces and stop words and concatenting the title and content together

For fine-tuning, I focused on mapping:
 - Article content → Summary
 - Since I didn’t have gold-standard summaries, I used the article title as a proxy summary.

The dataset was then tokenized using Hugging Face’s AutoTokenizer and converted into a datasets.Dataset object for training.

### 2. Training DistilBART

I fine-tuned sshleifer/distilbart-cnn-12-6, a distilled version of BART specialized for summarization.
 - Max input length: 512 (for article content)
 - Max output length: 256 (for summaries, i.e. titles)
 - Optimizer, learning rate, and batch sizes were left mostly at default Hugging Face Trainer settings.
 - Metrics: I used ROUGE (ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum).

### 3. Results

| Epoch | Training Loss | Validation Loss | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
|-------|---------------|-----------------|---------|---------|---------|------------|
|   1   |     0.0521    |      0.1214     |  1.25   |  0.00   |  1.25   |    1.25    |
|   2   |     0.0867    |      0.0687     |  0.44   |  0.00   |  0.44   |    0.44    |
|   3   |     0.0772    |      0.0687     |  0.00   |  0.00   |  0.00   |    0.00    |


And here’s a sample output:

* **Generated:**

  > _We Asked Cardiologists What They’d Never Use in Their Kitchens—and What to Use Instead - EatingWell’s Editorial Guidelines Published on August 30, 2025 Credit: Getty Images. EatingWell design. By Kristin Montemarano. We may receive compensation. We_

* **Reference (title):**

  > _We Asked Cardiologists What They’d Never Use in Their Kitchens—and What to Use Instead - EatingWell_


### 4 Interpreting the Results

At first glance, the numbers were confusing.
- **Training Loss** steadily decreased, suggesting the model was “learning.”
- **Validation Loss** also looked reasonable.
- **ROUGE**, however, was extremely low (close to zero after epoch 3) meaning target didn't match.

So why the mismatch?

-- . **Loss values (Training & Validation Loss)**:

Loss measures how well the model predicts token sequences. It does not directly measure summary quality.
- My model was likely learning to copy or predict long sequences.
- Loss values looked fine because the model wasn’t “wrong” in a token sense.

-- . **ROUGE scores** :

ROUGE compares generated summaries with references based on token overlap.
- Since my references were titles, which are short and concise, overlap was almost nonexistent.
- Even if the model generated coherent text, it didn’t match the reference well → low ROUGE.

**Why Loss and ROUGE Told Different Stories**

- Concatenating the title into the input content was mistake.
- Since the title was also the target (summary), the model could "cheat" by copying it from the input instead of learning to generate a proper summary.
- This led to artificially low loss (the model was “good” at repeating) but poor ROUGE scores (because the output didn’t align well with the actual concise summary format).
- In other words: Loss told me the model was predicting tokens confidently, but ROUGE revealed the model wasn’t actually summarizing.

### 5. Challenge: No Human-Written Summaries
One of the biggest challenges was the lack of human-written summaries. News articles usually have titles and content, but not proper summaries. This raised the question: what should the model learn to generate?

#### Why This Matters
- Summarization models need (article → summary) pairs.
- If the target is weak (e.g., a title), the model learns the wrong task.
- This explains why loss looked fine but ROUGE was poor — I was effectively training a headline generation model, not a summarizer.

#### Options Without Human Summaries

1. Extractive Summarization (Heuristics)
    - Use the first 2–3 sentences of each article as a pseudo-summary.
    - Works decently in news because leads often act as summaries.

2. Pseudo-Labels from Pretrained Models
    - Run a strong summarizer (e.g., bart-large-cnn) to generate synthetic summaries.
    - Fine-tune DistilBART on these → distillation approach.

3. Titles as Headline Generation (What I Tried)
    - Short, catchy, but not real summaries.
    - Explains my poor ROUGE results.
4. Manual or Crowdsourced Summaries
    - Create a small gold dataset (200–500 samples).
    - Even small amounts of high-quality data could drastically improve performance.

### 5. Key Takeaways

- **Loss ≠ summary quality**. Loss measures token prediction, while ROUGE measures semantic overlap with the reference summary.
- **Data setup matters**. Concatenating title + content introduced noise and let the model “cheat.”
- **Better inputs = better learning**. Use only content as input and title as the target. If extra context (like the headline) is needed, structure it clearly instead of concatenating.
- **ROUGE did its job**. It correctly showed the gap between token-level loss and summary quality.


### 6. Next Steps

My immediate plan:

1. Implement extractive pseudo-summaries using first 2–3 sentences.
2. Increase data sample size for better fine-tunning
3. Remove the title and content concatenation
4. Try pseudo-labeling with bart-large-cnn.
5. Re-run fine-tuning and compare ROUGE scores.
6. Document whether results improve and if the model starts producing actual summaries instead of noisy outputs.


Stay tuned — this is just the beginning of my journey into fine-tuning summarization models!
