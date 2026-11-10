# Web Scraping Extraction Report

- **Target Domain**: `books.toscrape.com`
- **Total Records Extracted**: 100
- **Number of Pages Traversed**: 5
- **Average Book Price**: £34.56
- **Min Price**: £10.16 | **Max Price**: £58.11
- **In Stock Ratio**: 100.0%

## Rating Distribution

|   star_rating |   count |
|--------------:|--------:|
|             1 |      22 |
|             2 |      19 |
|             3 |      22 |
|             4 |      18 |
|             5 |      19 |

## Data Schema

| Column | Type | Description |
|---|---|---|
| `book_id` | Integer | Unique identifier |
| `title` | String | Full book title |
| `price_gbp` | Float | Price in British Pounds (£) |
| `star_rating` | Integer | Rating scale 1-5 |
| `availability` | String | Inventory availability |
| `upc` | String | Universal Product Code |
| `category` | String | Genre / Book category |
| `product_url` | String | Source URL |
| `image_url` | String | Cover image URL |
