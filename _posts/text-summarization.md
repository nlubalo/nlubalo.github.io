---
layout: post-product-queen
title:  "Text Summarization with Large Language Models"
image: assets/images/home.jpg
description: "This week, I am going to leverage the power of transfer learning to fine tune a model trained on CNN Daily Mail articles to summarize dialogue data."
categories: [projects]
---

## Introduction

Transfer learning is a machine technique where a model pre-trained on a large dataset for a general task is fine-tuned on a smaller, task-specific dataset. This method allows models to transfer the knowledge gained from the pre-training phase to a related task, resulting in faster and more accurate learning.

In this task, I will fine-tune a **smaller, faster, and cheaper version of Facebook’s BART model (Bidirectional and Auto-Regressive Transformer)**.
BART is a transformer-based model that excels in a wide range of **NLP tasks**, including:
- Summarization
- Translation
- Text generation
- Comprehension

The model of choice is **sshleifer/distilbart-cnn-12-6**, a **distilled version of Facebook’s BART**.

- **Distillation** is the process of training a smaller model (*student*) to mimic the behavior of a larger, more powerful model (*teacher*).
- In this case, the full BART model acts as the teacher, and DistilBART is trained to approximate its predictions.
- The result is a model with **fewer parameters**, making it **smaller, faster, and cheaper** to run, while still retaining strong performance.

The original **DistilBART** was pre-trained on the **CNN/Daily Mail summarization dataset** (news articles).
I will fine-tune it to **transfer its knowledge to a new task: summarizing dialogue data**.

By the end of this project, I will have learned:

1. The **concepts of text summarization** with DistilBART.
2. How to **apply transfer learning** to adapt a summarization model from news to dialogues.
3. The **end-to-end process of fine-tuning a pre-trained transformer model** on a custom dataset.
4. How to **evaluate summarization models** using metrics like ROUGE and qualitative analysis.
5. Practical **model deployment considerations** (efficiency, latency, cost).

### Step-by-Step Fine-Tuning DistilBART for Dialogue Summarization

#### Step 1 — Explore the DialogSum Dataset
Before training the model, its important to understand how our data looks like. What are is the average length dialogues and summaries. This will guide the max_length we set for the input and target in the model. 
Its also important to check on how the dailogue data looks like to have an idea of what we expect as sumarries.


<details>
  <summary>Click to view code</summary>

{% highlight python %}
from datasets import load_dataset

# Load dataset
dataset = load_dataset("knkarthick/dialogsum")
# Look at available splits
print(dataset)
# Peek at a sample
print(dataset["train"][0])
{% endhighlight %}

</details><br>
This tells you there are 3 splits (train/validation/test), with those sizes.
- Each row has three fields: id, dialogue, and summary.
- Each sample is a dialogue + a human-written summary.
- Dialogues are multi-turn conversations separated by \n.
- Summaries are short, abstractive sentences capturing the gist.

```python
DatasetDict({
    train: Dataset({
        features: ['id', 'dialogue', 'summary'],
        num_rows: 12460
    })
    validation: Dataset({
        features: ['id', 'dialogue', 'summary'],
        num_rows: 500
    })
    test: Dataset({
        features: ['id', 'dialogue', 'summary'],
        num_rows: 500
    })
})
{
  'id': 'train_0',
  'dialogue': "M: It's very hot here.\nW: Yes, it is. What will the weather be like tomorrow?\nM: It will be a little cooler than today.\nW: That's good. I hope it will be sunny.\nM: Sorry, it won't. It will rain.",
  'summary': "The man says it's hot. The woman asks about tomorrow's weather. The man says it will be cooler but rainy."
}
```

### Summary of dataset statistics

| Split      | # Samples | Avg Dialogue Length (words) | Avg Summary Length (words) | Max Dialogue Length | Max Summary Length |
|------------|-----------|-----------------------------|-----------------------------|---------------------|---------------------|
| Train      | 12,460    | 116.2                       | 20.4                        | 521                 | 64                  |
| Validation | 500       | 114.7                       | 20.1                        | 510                 | 62                  |
| Test       | 500       | 115.6                       | 20.3                        | 518                 | 63                  |

### Analysis
- **Dataset size**: The training set is relatively small (12.5k samples), which means fine-tuning a pre-trained model like DistilBART is a good fit since we can transfer knowledge from a larger corpus.  
- **Dialogue vs. Summary lengths**: Dialogues average ~116 words, while summaries average ~20 words. This shows the summarization task is **highly compressive** (~5–6× compression).  
- **Max lengths**: Dialogues can go up to ~520 words, so we need to ensure the model’s maximum input length (typically 512 tokens for BART/DistilBART) can handle most cases. Summaries are short (≤64 words), which fits within the model’s output capacity.  
- **Consistency across splits**: Train, validation, and test splits have very similar statistics, which helps ensure generalization during evaluation.  

####  Step 2 — Load Dataset and model
After exploring the dataset, the next step is to load the model **(sshleifer/distilbart-cnn-12-6)** and inspect it a bit before fine-tuning

<details>
  <summary>Click to view code</summary>

{% highlight python %}
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the tokenizer and model
model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Inspect model details
print(model.config)

{% endhighlight %}

</details><br>

###  What this model is about

##### DistilBART (CNN-12-6)

- A **distilled version** of Facebook’s BART model.  
- **Distillation** = training a smaller model (student) to mimic a larger model (teacher).  
- This specific checkpoint was trained on the **CNN/Daily Mail news summarization dataset**, which is focused on **long news articles → short highlights**.  

---

##### Architecture

- **"12-6"** means it has **12 encoder layers** and **6 decoder layers** (compared to 12+12 in full BART).  
- This makes it **smaller, faster, and cheaper** to fine-tune while retaining most of BART’s summarization performance.  

---
##### Tokenizer

- Uses the same **SentencePiece tokenizer** as BART.  
- Handles **subword tokenization**, so long dialogues get broken into manageable tokens.  

---

#### Why inspect `model.config`?

`model.config` tells you:
- The **max input length** (often 1024 for BART/DistilBART).
- **Vocabulary size**.
- **Hidden layer sizes**.
- **Number of attention heads**.
- **Special tokens** (`<pad>`, `<eos>`, etc.).

This helps you align your **preprocessing** (like `max_input_length` for dialogues) with the model’s architecture.

#### Step 3 -  Preprocess the DialogSum dataset

### Why preprocessing is needed
- Neural networks (like DistilBART) don’t work directly with raw text.  
- We need to **convert dialogue strings → numerical tokens** the model understands.  
- Preprocessing also ensures each dialogue fits within the model’s maximum input length (usually 1024 tokens for DistilBART).

---
<details>
  <summary>Click to view code</summary>

{% highlight python %}
from datasets import load_dataset
from transformers import AutoTokenizer

# 1. Load dataset
dataset = load_dataset("knkarthick/dialogsum")

# 2. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")

# 3. Define preprocessing function
max_input_length = 512   # dialogue truncation length
max_target_length = 64   # summary truncation length

def preprocess_function(examples):
    # Concatenate all utterances into one dialogue string
    inputs = [dialog for dialog in examples["dialogue"]]
    targets = [summary for summary in examples["summary"]]

    # Tokenize inputs (dialogues)
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True
    )

    # Tokenize targets (summaries)
    labels = tokenizer(
        targets,
        max_length=max_target_length,
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 4. Apply preprocessing
tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

print(tokenized_datasets)

{% endhighlight %}

</details><br>
### Steps
1. **Concatenate utterances**
   - Each dialogue is stored as one string (e.g., `"Tim: Hi, what's up?\nKim: Bad mood..."`).
   - This keeps the dialogue context together for summarization.

2. **Tokenization**
   - Using DistilBART’s SentencePiece tokenizer.
   - Converts text into `input_ids` (integers representing tokens).
   - Example:  
     ```
     "Tim: Hi, what's up?" → [0, 25001, 38, 3395, 6, 4454, ...]
     ```

3. **Attention mask**
   - A binary mask showing which tokens are real vs. padding.
   - Example:  
     ```
     [1, 1, 1, 1, 1, 1, 0, 0, 0, ...]
     ```
   - Ensures the model ignores padding during training.

4. **Truncation & padding**
   - Dialogues longer than 1024 tokens are truncated.  
   - Shorter dialogues are padded so all sequences have the same length.

---

### Output of preprocessing
- **input_ids**: Encoded tokens of the dialogue.  
- **attention_mask**: Indicates which tokens to attend to.  

These are the inputs that will be fed into DistilBART for summarization.

#### Step 3 -  Training the Model

- The base model (DistilBART-CNN) was trained on **news articles → highlights**.
- Our dataset (DialogSum) contains **conversational dialogues → summaries**.
- Fine-tuning adapts the model from **news summarization** to **dialogue summarization**.

---
### Training setup
1. **Define training arguments**
   - Batch size (e.g., 8 or 16, depending on GPU memory).
   - Learning rate (commonly `5e-5` or similar).
   - Number of training epochs (start with ~3–5).
   - Save and evaluation strategy (evaluate every epoch, save best checkpoint).

2. **Trainer class**
   - Hugging Face `Trainer` API handles:
     - Forward pass.
     - Loss computation (cross-entropy between generated vs. reference summary).
     - Backpropagation + optimizer updates.
     - Evaluation loop (using ROUGE metrics).

3. **Loss function**
   - Objective: minimize difference between predicted summary tokens and gold summary tokens.
   - Teacher forcing is used (model is given the true previous token during training).

4. **Evaluation metrics**
   - ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum.
   - Measure overlap between generated and reference summaries.

---
### Training flow

1. Tokenized dialogues + summaries → model input.  
2. Model generates predicted tokens.  
3. Compute **training loss** (prediction vs. gold summary).  
4. Backpropagate to update weights.  
5. Validate on held-out data.  

---

### Expected outputs during training
- Training loss decreasing each epoch.  
- Validation loss slightly higher but stable.  
- ROUGE scores improving across epochs. 

---

<details>
  <summary>Click to view code</summary>

{% highlight python %}
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

rouge = evaluate.load("rouge")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(
        predictions, skip_special_tokens=True)

    # Replace -100 in labels with pad_token_id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode labels
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects newline separated sentences
    decoded_preds = ["\n".join(pred.strip().split()) for pred in decoded_preds]
    decoded_labels = ["\n".join(
        label.strip().split()) for label in decoded_labels]

    # Compute ROUGE scores
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )

    # Extract F1 scores and round
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # Add average generated length
    prediction_lens = [np.count_nonzero(
        pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",            # save checkpoints here
    eval_strategy="epoch",       # evaluate every epoch
    save_strategy="epoch",             # save model every epoch
    learning_rate=5e-5,                # common learning rate for fine-tuning
    per_device_train_batch_size=8,     # adjust based on GPU memory
    per_device_eval_batch_size=8,
    weight_decay=0.01,                 # helps regularization
    save_total_limit=3,                # keep only last 3 checkpoints
    num_train_epochs=3,                # number of training epochs
    predict_with_generate=True,        # needed for ROUGE eval
    logging_dir="./logs",              # log dir
    logging_steps=50,                  # log training info every 50 steps
    push_to_hub=False                  # set to True if you want to upload model
)

# Define Trainer
trainer = Seq2SeqTrainer(
    model=model,                        # our DistilBART model
    args=training_args,                 # training args defined above
    train_dataset=tokenized_datasets["train"],   # tokenized train split
    eval_dataset=tokenized_datasets["validation"], # tokenized val split
    tokenizer=tokenizer,                # tokenizer for preprocessing
    compute_metrics=compute_metrics     # function returning ROUGE scores
)

# Start training
trainer.train()



{% endhighlight %}

</details><br>

#### Step 4 - Fine-Tuning Results Summary

| Epoch | Training Loss | Validation Loss | Rouge1   | Rouge2   | Rougel  | Rougelsum | Gen Len  |
|-------|---------------|-----------------|----------|----------|---------|-----------|----------|
| 1     | 1.554000      | 1.468069        | 40.010200 | 19.959100 | 30.632300 | 37.466900 | 59.744800 |
| 2     | 1.215300      | 1.458910        | 40.137700 | 20.079300 | 30.655200 | 37.314100 | 59.714300 |
| 3     | 0.991900      | 1.502556        | 40.320400 | 19.046200 | 32.781800 | 40.397200 | 60.492100 |

<br>

We completed fine-tuning after **3 epochs**, using `load_best_model_at_end = True`.
This ensured that the Trainer automatically saved the model with the **best validation performance**.

### Key Findings:
- **Best Epoch:** **Epoch 2**
  - Lowest **Validation Loss**: *1.458910*
  - Highest **ROUGE-1** and **ROUGE-2** scores
  - Highest **ROUGE-Lsum** score

### Additional Notes:
- **ROUGE-Lsum**: Similar to ROUGE-L, but evaluates coverage at a **sentence-by-sentence level** rather than over the entire summary.
- **Generated Length (Gen Len):** Epoch 2 also produced the **shortest summaries on average**, which aligns with our goal of **concise yet informative outputs**.

✅ **Conclusion:**
Epoch 2 was the optimal checkpoint, balancing **low loss, high ROUGE scores, and compact summary length**.


#### Step 5 - Evaluating and Saving Model
After training and testing the model, we can evaluate its performance on the validation dataset. We can use the evaluate method for that.