# Task 1: Web Scraping Pipeline

## 📋 Overview
This task fulfills **CodeAlpha Data Analytics Internship - Task 1 (Web Scraping)**.
It demonstrates automated, multi-page data extraction from the e-commerce catalog [Books to Scrape](https://books.toscrape.com), followed by automated cleaning, data normalization, schema validation, tabular export (`CSV` and `JSON`), and exploratory summary analytics.

---

## 🛠️ Tech Stack & Libraries
- **Language**: Python 3.10+
- **HTTP Requests**: `requests` (with connection pooling and custom User-Agent headers)
- **HTML Parsing**: `BeautifulSoup4` (`bs4`)
- **Data Engineering**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`

---

## 🏗️ Architecture & Scraping Workflow
```
[ Target Website: books.toscrape.com ]
                 │
                 ▼
  [ Multi-Page Crawler (Session & Headers) ]
                 │
                 ▼
     [ DOM Extraction with BeautifulSoup ]
     ├─ Title & Canonical URLs
     ├─ Star Rating (Text-to-Numeric mapping)
     ├─ Price Cleaned (GBP float conversion)
     ├─ Stock Availability Status
     └─ Detail Page Parsing (UPC, Tax, Category)
                 │
                 ▼
        [ Pandas Pipeline ]
     ├─ Data Normalization & Cleaning
     ├─ Schema Validation
     └─ Unique Identifier Assignment
                 │
                 ├──────────────────────────┐
                 ▼                          ▼
     [ data/scraped_books.csv ]    [ data/scraped_books.json ]
                 │
                 ▼
[ outputs/scraping_summary.png & outputs/scraping_report.md ]
```

---

## 📊 Extracted Data Schema
The dataset contains structured attributes for each book:

| Column | Data Type | Description |
|---|---|---|
| `book_id` | Integer | Auto-incremented primary key |
| `title` | String | Full book title |
| `price_gbp` | Float | Price in British Pounds (£) |
| `star_rating` | Integer | Normalized rating from 1 to 5 |
| `availability` | String | Stock status (`In stock` / `Out of stock`) |
| `upc` | String | Universal Product Code / Hash ID |
| `category` | String | Genre / Department |
| `product_url` | String | Direct canonical URL to the product |
| `image_url` | String | Absolute cover image URL |

---

## 🚀 How to Run
From the root directory or task folder:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the scraping script
python Task1_Web_Scraping/web_scraper.py
```

---

## 📈 Visual Summary
The scraper automatically generates an analytical overview plot saved at `outputs/scraping_summary.png`:
- **Price Distribution**: Price frequency histogram with Mean price overlay.
- **Star Rating Breakdown**: Frequency count per star tier.
- **Price vs Rating**: Average price across 1 to 5-star ratings.
- **Category Composition**: Donut chart representing genres.

---

## 📄 Key Insights
- Catalog prices are uniformly distributed across the £10 to £60 range, indicating varied price tier offerings.
- Ratings show balanced distribution across tiers, ideal for downstream preference or recommendation systems.
- Robust exception handling ensures uninterrupted crawling even under network latency or missing attributes.
