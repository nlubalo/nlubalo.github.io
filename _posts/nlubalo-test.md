---
layout: post-product-queen
title:  "Nlubalo Test"
image: assets/images/home.jpg
description: "A detailed overview of a project where I designed and built a new data pipeline using modern cloud technologies to support real-time analytics."
categories: [projects]
---

<p>
I'm a data pro fluent in scalable pipelines, cloud platforms, Agent AI, and LLM-powered analytics. I turn raw data into intelligent systems that learn, adapt, and deliver insights at scale.
</p>
<p>
    From real-time streaming to bulletproof ETL, from AI-first architectures, my expertise thrives at the intersection of data engineering, AI, and strategy. I’m obsessed with making complex data ecosystems run seamlessly — and turning chaos into clarity.
</p>

---

## The Challenge

<p>
    Our primary challenge was to migrate a legacy, batch-based data system to a real-time, scalable architecture to support on-demand analytics. The existing system was slow, prone to failure, and could not handle the growing volume of data.
</p>
<img src="{{site.baseurl}}/assets/images/nlubalo.jpeg" class="img-fluid rounded" alt="Diagram of the old data architecture">

## My Solution

<p>
    I designed and built a new data pipeline using a modern stack. The solution leverages cloud-native services to handle streaming data, a data lake for storage, and automated ETL processes.
</p>

### Technologies Used

<ul>
    <li>Python</li>
    <li>Apache Spark</li>
    <li>AWS S3</li>
    <li>Amazon Redshift</li>
</ul>

### Code Snippet

<p>
Here is a small Python function that shows part of the data transformation logic.
</p>

```python
import pandas as pd

def transform_data(raw_data):
  """Transforms raw data into a clean DataFrame."""
  df = pd.DataFrame(raw_data)
  df['timestamp'] = pd.to_datetime(df['timestamp'])
  df = df.dropna()
  return df
  ```