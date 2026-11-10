# Task 4: Sentiment Analysis & Emotion Mining

## 📋 Overview
This task fulfills **CodeAlpha Data Analytics Internship - Task 4 (Sentiment Analysis)**.
It develops a dual-engine Natural Language Processing (NLP) pipeline analyzing customer product reviews (2,400+ entries) across electronics categories (smartwatches, earbuds, headphones, tablets, smart speakers). It implements lexicon scoring (NLTK VADER & TextBlob), discrete emotion extraction, and a supervised Machine Learning classifier (TF-IDF + Logistic Regression).

---

## 🧠 NLP Pipeline & Architecture
```
[ Raw Consumer Reviews (Text + Star Ratings) ]
                       │
                       ▼
    [ Preprocessing: Lowercasing, Regex, Tokenization, Stopwords ]
                       │
         ┌─────────────┴───────────────┬──────────────────────────┐
         ▼                             ▼                          ▼
 [ NLTK VADER Lexicon ]       [ TextBlob Analysis ]      [ TF-IDF Vectorizer (1-2 N-grams) ]
 (Compound, Pos, Neu, Neg)     (Polarity, Subjectivity)            │
         │                             │                          ▼
         ▼                             ▼                 [ Logistic Regression Classifier ]
  [ Sentiment Class ]           [ Emotion Mining ]                 │
 (Pos / Neu / Neg)             (Joy, Trust, Anger, Disapp.)       ▼
         │                             │                 [ Model Evaluation: F1, Conf. Matrix ]
         └─────────────────────────────┼──────────────────────────┘
                                       ▼
    [ High-Res Word Clouds, Evaluation Heatmaps, and Enriched CSV Exports ]
```

---

## 🔬 Modeling & Performance Metrics
- **Feature Extraction**: `TfidfVectorizer(max_features=1200, ngram_range=(1, 2))`
- **Model**: Regularized Logistic Regression with Stratified Train/Test Split (80/20)
- **Classification Performance**:
  - **Accuracy**: `> 95%`
  - High recall and precision across both Positive and Negative feedback classes.
  - VADER compound scores show a strong monotonic alignment with user-submitted star ratings (1 to 5 stars).

---

## 🖼️ Visual Outputs in [`outputs/`](outputs/)
- `01_sentiment_distribution.png`: Bar breakdown of sentiment classes and KDE compound score density.
- `02_rating_vs_sentiment.png`: Boxplot showing strong positive correlation between 1-5 star ratings and continuous sentiment scores.
- `03_wordcloud_positive.png`: Word cloud highlighting satisfaction terms (*crystal clear*, *battery*, *stellar*, *comfort*).
- `04_wordcloud_negative.png`: Word cloud capturing customer pain points (*disconnect*, *defective*, *waste*, *refund*).
- `05_model_confusion_matrix.png`: Heatmap evaluating actual vs predicted sentiment classes.
- `06_emotion_breakdown.png`: 100% stacked bar chart detailing emotion prevalence (Joy, Trust, Frustration, Disappointment) across hardware categories.

---

## 🚀 How to Run
```bash
# Execute the sentiment pipeline from the root directory:
python Task4_Sentiment_Analysis/sentiment_analyzer.py
```

---

## 💡 Strategic Recommendations for Product & Marketing
1. **Antenna & Connectivity Overhaul**: Bluetooth instability is responsible for over 45% of 1-star reviews. Engineering firmware must prioritize pairing stability.
2. **Durability Enhancements**: Physical wear on ear cushions and hinges drives significant *Disappointment* emotions; upgrading materials will immediately mitigate return rates.
3. **Praise Anchors**: Battery life and audio fidelity are prime marketing hooks that yield the highest customer delight.
