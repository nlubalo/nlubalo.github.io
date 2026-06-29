---
layout: post-product-queen
title:  "Customer Analytics Week 1: Descriptive Analytics"
image: assets/images/home.jpg
tags: [sticky]
categories: [customer-analytics]
---
I recently started the Wharton Customer Analytics course on Coursera. This post documnets my week 1 learnings on descriptive analytics, the foundation of the entire analytics journey alongside a hands-on Python project I'm building in parallel.

---
## What is Descriptive Analytics
Descriptive analytics is the practice of collecting and summarizing historical customer data to understand what has happened in a business. Rather than predicting the future or recommending actions, it focuses on turning raw data into a clear picture giving decision-makers a solid factual base to wok from

Think of it as answering the question: "What happened, and to whom?"

---

## Key Concepts

### Data Collection Methods
Descriptive analytics startes with gathering the right data. Common methods include:

- **Surveys**: Direct feedback from customers about their experiences and preferences
- **Customer loyalty programs** : transactional and behaviour data tied to the identifiable individuals
- **Transaction records**: Purchase history, order values, frequency and product preferences
- **Passive data collection**: Online ativity tractking such as page views, click paths and session durations


Each method captures a different slice of customer behaviour and combining them gives a richer picture than any single source alone

---

## Purpose and Applications

Once data is collected, descriptive analytics helps businesses:

- **Identify trends** - e.g revenue growth over time, seasonal spikes in demand
- **Spot patterns** - e.g which product are frequently bought together
- **Segment customers** - grouping customers by behavior, value or demographics to tailor strategies
- **Media planning effectiveness** - analyzing which channels and campagigns droves the most customer engagement or conversions

Two practical examples from the course stood out:

- **Net Promoter Score (NPS)** — a survey-based metric that summarizes customer satisfaction and loyalty in a single number. It's derived from one simple question. **On a scale of 1 to 10 How likely
are you to recommend this service/product/company to a friend?** People who say 9 or 10 are called **promoters**, those who say 7 or 8 are **passive** and those who say 0 to 6 are **detractors**
The Net Promoter Score is calculated by subtracting the percentage of detractors from the percentage of promoters. This score helps businesses understand how happy their customers are and how strong their brand is.
- **Media planning effectiveness** — analyzing which channels and campaigns drove the most customer engagement or conversions


---

## Benefits and Limitations

### What descriptive analytics does well
- Provides an objective, data-driven account of what actually happened
- Makes comples data accessible through summaries, visualizations and key metrics
- Enables businnesses to move away from gut-feel decisions towards evidence-based ones

### Where it falls short
- It looks **backwards** not forward, it tells you what happened not what will happen
- It does **not prescribe actions** - It surfaces insights but stops short of recommending what to do next
- On its own, it cannot establish **causation**  - Patterns in historical data may be correlations not cause

---

## Where it Fits in the Analytics Journey

Descriptive analytics is the **firrst of three layers**

| Layer | Question it answers | Example |
|---|---|---|
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

---

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

This split directly informs CLV modeling in Week 3 — repeat buyers are the foundation of long-term customer value.


### Revenue Concentration (Pareto Analysis)

Do a small number of customers drive most of the revenue?

```python
customer_revenue = df.groupby('CustomerID')['TotalPrice'].sum().sort_values(ascending=False)
cumulative = customer_revenue.cumsum() / customer_revenue.sum() * 100
```

![Pareto Chart](https://raw.githubusercontent.com/nlubalo/business_analytics/main/customer_analytics/data/charts/pareto_chart.png)

The Pareto principle holds — a small fraction of customers account for the majority of revenue. This has direct implications for where to focus retention and loyalty efforts.

---

### Geographic Breakdown

```python
country_revenue = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10)
```

![Revenue by Country](https://raw.githubusercontent.com/nlubalo/business_analytics/main/customer_analytics/data/charts/country_revenue.png)

The UK dominates — as expected for a domestic gift retailer — but international markets are worth watching for growth opportunities.

---

### Correlation vs. Causation in Practice

A key Week 1 concept applied: order frequency and total spend are correlated, but that doesn't mean ordering more *causes* higher spend. The causality could run the other way (high-value customers naturally order more), or a third factor (loyalty program membership) could drive both.

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

**Next up — Week 2 & 3:** RFM scoring and Customer Lifetime Value (CLV) modeling on the same dataset.

> 📓 **Full notebook on GitHub:** [descriptive_analytics.ipynb](https://github.com/nlubalo/business_analytics/blob/main/customer_analytics/descriptive_analytics.ipynb)

---

*This post is part of my series documenting my journey through the Wharton Customer Analytics course on Coursera, alongside a hands-on Python project.*
