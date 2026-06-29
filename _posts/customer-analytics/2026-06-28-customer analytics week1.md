---
layout: post-product-queen
title:  "Customer Analytics Week 1: Descriptive Analytics"
image: assets/images/home.jpg
tags: [sticky]
categories: [customer-analytics]
description: "Week 1 of the Wharton Customer Analytics course: descriptive analytics foundations and a hands-on Python project on the UCI Online Retail dataset."
---

> 📓 **Hands-on notebook:** [descriptive_analytics.ipynb](https://github.com/nlubalo/business_analytics/blob/main/customer_analytics/descriptive_analytics.ipynb) — UCI Online Retail dataset (~500K rows)

I recently started the Wharton Customer Analytics course on Coursera. This post documents my Week 1 learnings on descriptive analytics — the foundation of the entire analytics journey — alongside a hands-on Python project I'm building in parallel.

---

## What is Descriptive Analytics

Descriptive analytics is the practice of collecting and summarising historical customer data to understand what has happened in a business. Rather than predicting the future or recommending actions, it focuses on turning raw data into a clear picture, giving decision-makers a solid factual base to work from.

Think of it as answering the question: **"What happened, and to whom?"**

## Key Concepts

### Data Collection Methods

Descriptive analytics starts with gathering the right data. Common methods include:

- **Surveys** — direct feedback from customers about their experiences and preferences
- **Customer loyalty programs** — transactional and behavioural data tied to identifiable individuals
- **Transaction records** — purchase history, order values, frequency, and product preferences
- **Passive data collection** — online activity tracking such as page views, click paths, and session durations

Each method captures a different slice of customer behaviour. Combining them gives a richer picture than any single source alone.

## Purpose and Applications

Once data is collected, descriptive analytics helps businesses:

- **Identify trends** — e.g. revenue growth over time, seasonal spikes in demand
- **Spot patterns** — e.g. which products are frequently bought together
- **Segment customers** — grouping by behaviour, value, or demographics to tailor strategies
- **Measure media effectiveness** — analysing which channels and campaigns drove the most engagement or conversions

### Net Promoter Score (NPS)

One of the course's standout examples. NPS is a survey-based metric that summarises customer satisfaction and loyalty in a single number, derived from one question:

> **On a scale of 1 to 10, how likely are you to recommend this product/service to a friend?**

| Score | Label |
| --- | --- |
| 9 - 10 | **Promoters** |
| 7 - 8 | **Passives** |
| 0 - 6 | **Detractors** |

**NPS = % Promoters − % Detractors**

This score gives businesses a quick read on brand health and customer loyalty that's easy to track consistently over time.

## Benefits and Limitations

### What descriptive analytics does well

- Provides an objective, data-driven account of what actually happened
- Makes complex data accessible through summaries, visualisations, and key metrics
- Enables businesses to move from gut-feel decisions to evidence-based ones

### Where it falls short

- It looks **backwards**, not forward — it tells you what happened, not what will happen
- It does **not prescribe actions** — it surfaces insights but stops short of recommending what to do next
- It cannot establish **causation** — patterns in historical data may be correlations, not causes

## Where it Fits in the Analytics Journey

Descriptive analytics is the **first of three layers**:

| Layer | Question it answers | Example |
| --- | --- | --- |
| **Descriptive** | What happened? | Monthly revenue by customer segment |
| **Predictive** | What will happen? | Which customers are likely to churn? |
| **Prescriptive** | What should we do? | Which customers should we target with a retention offer? |

Each layer builds on the previous one. Without solid descriptive work, predictive models have no reliable foundation — and prescriptive recommendations risk being built on guesswork.

---

## Hands-On: Python Project — UCI Online Retail Dataset

To put these concepts into practice, I applied them to the [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail) — a real e-commerce transaction dataset with ~500K rows covering a UK-based gift retailer.

### Data Cleaning

Before any analysis, I cleaned the raw data: removing transactions with missing customer IDs, cancellations (invoices starting with `C`), and rows with negative quantities or prices. I also engineered a `TotalPrice` column.

```python
import pandas as pd

df = pd.read_excel('OnlineRetail.xlsx', dtype={'CustomerID': str})

# Remove missing CustomerIDs
df = df.dropna(subset=['CustomerID'])

# Remove cancellations
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

# Remove bad quantities/prices
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# Engineer TotalPrice
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
```

### Revenue Trends Over Time

The first question: is revenue growing, declining, or seasonal?

```python
monthly_revenue = df.groupby(df['InvoiceDate'].dt.to_period('M'))['TotalPrice'].sum()
monthly_revenue.plot(kind='line', marker='o', title='Monthly Revenue Over Time')
```

![Monthly Revenue Trend](https://raw.githubusercontent.com/nlubalo/business_analytics/main/customer_analytics/data/charts/revenue_trend.png)

A clear Q4 spike is visible — consistent with a gift retailer's seasonal pattern. Importantly, this is a **correlation** with the holiday season, not proof that any particular campaign caused the spike.

### Repeat vs. One-Time Buyers

A key CLV question: what share of customers come back?

```python
order_counts = df.groupby('CustomerID')['InvoiceNo'].nunique()
repeat_buyers = (order_counts > 1).sum()
one_time_buyers = (order_counts == 1).sum()
```

![Repeat vs One-Time Buyers](https://raw.githubusercontent.com/nlubalo/business_analytics/main/customer_analytics/data/charts/customer_behavior.png)

This split directly informs CLV modelling in Week 3 — repeat buyers are the foundation of long-term customer value.

### Revenue Concentration (Pareto Analysis)

Do a small number of customers drive most of the revenue?

```python
customer_revenue = df.groupby('CustomerID')['TotalPrice'].sum().sort_values(ascending=False)
cumulative = customer_revenue.cumsum() / customer_revenue.sum() * 100
```

![Pareto Chart](https://raw.githubusercontent.com/nlubalo/business_analytics/main/customer_analytics/data/charts/pareto_chart.png)

The Pareto principle holds — a small fraction of customers account for the majority of revenue. This has direct implications for where to focus retention and loyalty efforts.

### Geographic Breakdown

```python
country_revenue = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10)
```

![Revenue by Country](https://raw.githubusercontent.com/nlubalo/business_analytics/main/customer_analytics/data/charts/country_revenue.png)

The UK dominates — as expected for a domestic gift retailer — but international markets are worth watching for growth opportunities.

### Correlation vs. Causation in Practice

A key Week 1 concept: order frequency and total spend are correlated, but that doesn't mean ordering more *causes* higher spend. The causality could run the other way (high-value customers naturally order more), or a third factor like loyalty program membership could drive both.

```python
customer_summary = df.groupby('CustomerID').agg(
    TotalSpend=('TotalPrice', 'sum'),
    NumOrders=('InvoiceNo', 'nunique')
)
```

![Frequency vs Spend](https://raw.githubusercontent.com/nlubalo/business_analytics/main/customer_analytics/data/charts/freq_vs_spend.png)

---

## My Takeaway

Week 1 reframed how I think about data work. It's tempting to jump straight to models and predictions, but the discipline of thoroughly understanding historical data first is what separates rigorous analytics from noise. Descriptive analytics isn't a lesser form of analysis — it's the bedrock everything else is built on.

**Next up — Week 2 & 3:** RFM scoring and Customer Lifetime Value (CLV) modelling on the same dataset.
