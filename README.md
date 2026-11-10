# 📊 CodeAlpha Data Analytics Internship — Complete Portfolio

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Libraries-Pandas%20%7C%20Seaborn%20%7C%20Scikit--Learn%20%7C%20NLTK-orange.svg)](#-tech-stack--tools)
[![Internship](https://img.shields.io/badge/CodeAlpha-Data%20Analytics-red.svg)](https://www.codealpha.tech/)
[![Status](https://img.shields.io/badge/Status-Completed%20(4%2F4%20Tasks)-brightgreen.svg)](#-task-summary)

This repository contains the complete, production-grade project deliverables for the **CodeAlpha Data Analytics Internship Program**. While interns were tasked to complete any 2 or 3 of the 4 assigned tasks, **this portfolio completes all 4 tasks (100% curriculum completion)** with real datasets, reproducible scripts, statistical hypothesis testing, machine learning pipelines, and executive dashboards.

---

## 📑 Table of Contents
1. [Internship Overview & Objectives](#-internship-overview--objectives)
2. [Task Summary Matrix](#-task-summary-matrix)
3. [Repository Directory Structure](#-repository-directory-structure)
4. [Tech Stack & Tools](#-tech-stack--tools)
5. [In-Depth Task Breakdown](#-in-depth-task-breakdown)
   - [Task 1: Web Scraping](#-task-1-web-scraping)
   - [Task 2: Exploratory Data Analysis (EDA)](#-task-2-exploratory-data-analysis-eda)
   - [Task 3: Data Visualization & Storytelling](#-task-3-data-visualization--storytelling)
   - [Task 4: Sentiment Analysis & Emotion Mining](#-task-4-sentiment-analysis--emotion-mining)
6. [Quickstart & Reproduction Guide](#-quickstart--reproduction-guide)
7. [LinkedIn Video Presentation Guide](#-linkedin-video-presentation-guide)
8. [CodeAlpha Submission Verification Checklist](#-codealpha-submission-verification-checklist)

---

## 🏢 Internship Overview & Objectives
The **CodeAlpha Data Analytics Internship** provides hands-on exposure to data extraction, statistical interpretation, exploratory analysis, visual storytelling, and natural language processing. 

### Key Deliverables Completed:
- ✅ **Automated Web Scraping**: Resilient crawling of public e-commerce data with schema normalization.
- ✅ **Hypothesis-Driven EDA**: Statistical hypothesis testing (ANOVA, Welch's t-test, Chi-square), outlier detection, and discount margin analysis.
- ✅ **Publication-Grade Visualization**: Custom-styled multi-panel executive dashboards and data stories.
- ✅ **NLP & Sentiment Classification**: Lexicon analysis (VADER, TextBlob), discrete emotion detection, and supervised Machine Learning (TF-IDF + Logistic Regression).

---

## 🏆 Task Summary Matrix

| Task | Domain | Primary Methodology | Key Libraries | Primary Output |
|---|---|---|---|---|
| **Task 1: Web Scraping** | E-Commerce Books Catalog | Multi-page DOM parsing, regex cleaning, schema validation | `requests`, `BeautifulSoup4`, `pandas` | [scraped_books.csv](Task1_Web_Scraping/data/scraped_books.csv), [Summary Plot](Task1_Web_Scraping/outputs/scraping_summary.png) |
| **Task 2: Exploratory Data Analysis (EDA)** | Superstore Retail Sales | Missingness audit, parametric & non-parametric hypothesis testing | `pandas`, `scipy.stats`, `seaborn` | [6 Analytical Plots](Task2_Exploratory_Data_Analysis/outputs/), [Executive Report](Task2_Exploratory_Data_Analysis/outputs/eda_executive_report.md) |
| **Task 3: Data Visualization** | Global Streaming Media | GridSpec multi-panel executive dashboard, bubble regression, trendlines | `matplotlib`, `seaborn`, `numpy` | [5 High-Res Dashboards](Task3_Data_Visualization/outputs/), [Data Story Narrative](Task3_Data_Visualization/outputs/data_story_narrative.md) |
| **Task 4: Sentiment Analysis** | Consumer Tech Reviews | VADER lexicon, TextBlob, TF-IDF + Logistic Regression, WordClouds | `nltk`, `scikit-learn`, `wordcloud` | [Confusion Matrix Heatmap](Task4_Sentiment_Analysis/outputs/05_model_confusion_matrix.png), [Enriched Data](Task4_Sentiment_Analysis/outputs/processed_sentiment_results.csv) |

---

## 📂 Repository Directory Structure

```plaintext
CodeAlpha_DataAnalytics/
├── README.md                                 # Master Portfolio Documentation
├── requirements.txt                          # Pinned Python Dependencies
├── .gitignore                                # Git Ignore Rules (Python, Data, Caches)
│
├── Task1_Web_Scraping/
│   ├── README.md                             # Task 1 Documentation & Selectors
│   ├── web_scraper.py                        # Automated Web Scraping Script
│   ├── data/
│   │   ├── scraped_books.csv                 # Clean Extracted CSV Dataset
│   │   └── scraped_books.json                # Structured JSON Dataset
│   └── outputs/
│       ├── scraping_summary.png              # Analytical Overview Figure
│       └── scraping_report.md                # Data Extraction Audit Report
│
├── Task2_Exploratory_Data_Analysis/
│   ├── README.md                             # EDA Questions, Hypotheses & Methodology
│   ├── eda_analysis.py                       # End-to-End Statistical EDA Pipeline
│   ├── data/
│   │   └── superstore_sales.csv              # Retail Transactions Dataset (3,000+ rows)
│   └── outputs/
│       ├── 01_missing_values_matrix.png      # Missingness Heatmap
│       ├── 02_sales_profit_distribution.png  # Skewness & Outliers
│       ├── 03_correlation_heatmap.png        # Feature Correlation Matrix
│       ├── 04_category_performance.png       # Gross Sales vs Net Profit
│       ├── 05_discount_impact_profit.png     # Discount Margin Erosion Boxplot & Loss Curve
│       ├── 06_regional_insights.png          # Regional Performance & Customer Segments
│       └── eda_executive_report.md           # Formal Business Report with p-values
│
├── Task3_Data_Visualization/
│   ├── README.md                             # Visualization Design Principles & Insights
│   ├── data_visualizer.py                    # Multi-Panel Dashboard & Storytelling Script
│   ├── data/
│   │   └── streaming_platform_data.csv       # Streaming Catalog Dataset (2,500 rows)
│   └── outputs/
│       ├── 01_executive_dashboard.png        # Integrated Executive Command Dashboard
│       ├── 02_genre_popularity_distribution.png # Violin + Box Quartile Plots
│       ├── 03_ratings_vs_duration_bubble.png # Runtime vs IMDb Bubble Scatter
│       ├── 04_content_release_timeline.png   # 20-Year Volume vs Quality Dual-Axis Chart
│       ├── 05_content_type_donut.png         # Format Split & Genre Pareto Bar
│       └── data_story_narrative.md           # The Quality-Scale Paradox Narrative
│
└── Task4_Sentiment_Analysis/
    ├── README.md                             # NLP Architecture, Evaluation & Lexicons
    ├── sentiment_analyzer.py                 # Full NLP, ML & Emotion Mining Pipeline
    ├── data/
    │   └── customer_reviews.csv              # Consumer Reviews Dataset (2,400 rows)
    └── outputs/
        ├── 01_sentiment_distribution.png     # Class Share & Polarity Density
        ├── 02_rating_vs_sentiment.png        # Star Rating vs VADER Compound Score
        ├── 03_wordcloud_positive.png         # High-Frequency Praise Vocabulary
        ├── 04_wordcloud_negative.png         # Pain Points & Friction Lexicon
        ├── 05_model_confusion_matrix.png     # Supervised Classifier Confusion Matrix
        ├── 06_emotion_breakdown.png          # Emotion Distribution Across Categories
        ├── processed_sentiment_results.csv   # Scored & Classified Dataset
        └── sentiment_business_insights.md    # Product & Marketing Strategic Plan
```

---

## 🛠️ Tech Stack & Tools

- **Core Language**: Python 3.10+
- **Data Manipulation**: `pandas`, `numpy`
- **Web Scraping**: `requests`, `BeautifulSoup4` (`bs4`), `lxml`
- **Statistical Modeling**: `scipy.stats` (One-way ANOVA, Welch's t-test, Chi-Square contingency)
- **Data Visualization**: `matplotlib`, `seaborn`, `wordcloud`
- **Natural Language Processing**: `nltk` (VADER, stopwords, tokenization), `textblob`
- **Machine Learning**: `scikit-learn` (`TfidfVectorizer`, `LogisticRegression`, `classification_report`)

---

## 🔍 In-Depth Task Breakdown

### 🕷️ Task 1: Web Scraping
- **Target Source**: `https://books.toscrape.com`
- **Key Capabilities**:
  - Traverses paginated catalog structures using persistent HTTP sessions.
  - Extracts titles, prices (cleaned as float), star ratings (converted from word classes to numeric 1-5), inventory status, category breadcrumbs, UPC codes, and image links.
  - Generates comprehensive data validation exports ([CSV](Task1_Web_Scraping/data/scraped_books.csv), [JSON](Task1_Web_Scraping/data/scraped_books.json)) and a visual extraction summary.

### 📈 Task 2: Exploratory Data Analysis (EDA)
- **Dataset**: Superstore Retail Orders (3,005 transactions, 14 attributes).
- **Core Business Questions & Statistical Tests**:
  1. *Do product categories differ significantly in profit margins?*
     - **One-way ANOVA**: $F = 219.55$, $p = 1.025 \times 10^{-89}$. Reject $H_0$; Technology significantly outperforms Furniture.
  2. *What is the impact of heavy discounting ($>20\%$)?*
     - **Welch's t-test**: $t = -19.44$, $p = 9.38 \times 10^{-70}$. Confirms steep discount erosion where mean profit plunges to $-\$230.98$.
  3. *Is customer segment associated with preferred shipping method?*
     - **Chi-Square Test**: $\chi^2 = 3.93$, $p = 0.686$ ($DoF = 6$). No statistically significant dependency detected.
- **Data Quality Audit**: Identifies and remediates missing data in shipping costs and customer segments using median imputation and mode assignment.

### 🎨 Task 3: Data Visualization & Storytelling
- **Theme**: Global Streaming Wars & The Quality-Scale Paradox (2,500 catalog titles).
- **Visual Design**:
  - **Integrated Executive Dashboard**: Multi-panel view combining market share donuts, 20-year production areas, regional hub horizontal bars, and IMDb boxplots.
  - **Quality vs Runtime Bubble Chart**: OLS regression trendline overlay showing that films between 95 and 115 minutes achieve peak viewer reception.
  - **Volume Expansion vs Quality Timeline**: Demonstrates how platforms scaled title acquisitions post-2015 while maintaining stable audience ratings (~6.5).

### 💬 Task 4: Sentiment Analysis & Emotion Mining
- **Dataset**: Consumer Electronics Customer Feedback (2,400 reviews).
- **Dual Engine Architecture**:
  - **Lexicon Engine**: Computes continuous polarity with NLTK VADER and TextBlob.
  - **Emotion Classifier**: Categorizes feedback into Joy/Delight, Trust, Frustration, Disappointment, and Neutral states.
  - **Supervised ML Model**: TF-IDF n-gram vectorization with Logistic Regression achieving high test accuracy.
  - **Visual Diagnostics**: Generates side-by-side positive/negative word clouds and evaluation heatmaps.

---

## ⚡ Quickstart & Reproduction Guide

### 1. Clone the Repository
```bash
git clone https://github.com/shivansh2277/codealpha_tasks.git
cd codealpha_tasks
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Run Any Task Pipeline Individually
```bash
# Task 1: Web Scraping
python Task1_Web_Scraping/web_scraper.py

# Task 2: Exploratory Data Analysis
python Task2_Exploratory_Data_Analysis/eda_analysis.py

# Task 3: Data Visualization
python Task3_Data_Visualization/data_visualizer.py

# Task 4: Sentiment Analysis
python Task4_Sentiment_Analysis/sentiment_analyzer.py
```

---

## 🎥 LinkedIn Video Presentation Guide

As instructed in the CodeAlpha guidelines (*"Post a video explanation of your project on LinkedIn with GitHub repo link"*), here is a suggested 2-minute recording outline:

1. **Introduction (0:00 - 0:20)**:
   - Introduce yourself and mention the **CodeAlpha Data Analytics Internship**.
   - Highlight that you completed all 4 tasks covering Web Scraping, EDA, Data Visualization, and Sentiment Analysis.
2. **Task 1 & Task 2 Walkthrough (0:20 - 0:50)**:
   - Screen-share `scraping_summary.png` and explain how multi-page crawling and schema validation work.
   - Show `05_discount_impact_profit.png` from Task 2 and explain the Welch's t-test proving how discounts above 20% destroy retail margins.
3. **Task 3 & Task 4 Walkthrough (0:50 - 1:30)**:
   - Present `01_executive_dashboard.png` from Task 3 and discuss the data story on streaming platform dynamics.
   - Show `03_wordcloud_positive.png` and the confusion matrix from Task 4, explaining the VADER + TF-IDF classification and emotion mining.
4. **Conclusion & GitHub Link (1:30 - 2:00)**:
   - Summarize key learnings in Python, statistics, and machine learning.
   - Tag **@CodeAlpha** and invite viewers to check out your GitHub repository link: `https://github.com/shivansh2277/codealpha_tasks`.

---

## ✅ CodeAlpha Submission Verification Checklist

- [x] Repository named in accordance with CodeAlpha instructions: `codealpha_tasks`
- [x] All source code cleanly structured and documented with docstrings
- [x] Completed at least 2-3 tasks (Exceeded: **Completed all 4 of 4 tasks**)
- [x] Datasets included in `data/` subdirectories
- [x] High-resolution charts exported in `outputs/` subdirectories
- [x] Clear `requirements.txt` and `.gitignore` included
- [x] Detailed task-level READMEs in each task folder
- [x] Ready for GitHub deployment and LinkedIn video explanation

---

## 📬 Contact & Connect
- **Author**: Shivansh
- **GitHub**: [@shivansh2277](https://github.com/shivansh2277)
- **Organization**: [CodeAlpha](https://www.codealpha.tech/)
