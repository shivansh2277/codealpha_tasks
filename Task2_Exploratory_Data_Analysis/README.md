# Task 2: Exploratory Data Analysis (EDA)

## 📋 Overview
This task fulfills **CodeAlpha Data Analytics Internship - Task 2 (Exploratory Data Analysis)**.
It demonstrates an enterprise-grade exploratory data analysis performed on a retail superstore transaction dataset (3,000+ orders), rigorously exploring data distributions, correlations, outliers, missingness, statistical hypothesis testing, and business recommendations.

---

## 🎯 Pre-Analysis Questions Formulated
Before touching the data, the following core business questions were established:
1. **Revenue vs. Profitability**: Which product categories and sub-categories are true profit engines, and which are unprofitable loss-leaders?
2. **Discount Thresholds**: At what exact discount percentage does an order flip from profitable to net-negative?
3. **Customer Segment Economics**: Do Corporate, Consumer, and Home Office clients exhibit distinct purchasing behaviors and average order values (AOV)?
4. **Regional Disparities**: Are regional profit variances statistically significant, or driven purely by random sampling noise?
5. **Operational Logistics**: How do shipping modes (Same Day vs First/Second/Standard Class) affect delivery lead times and profitability?

---

## 🔬 Statistical Hypothesis Testing
To validate assumptions with statistical rigor:

### 1. One-Way ANOVA: Category Profitability
- **$H_0$**: $\mu_{\text{Furniture}} = \mu_{\text{Technology}} = \mu_{\text{Office Supplies}}$
- **Test**: One-way ANOVA ($F$-statistic)
- **Result**: $p < 0.001$, firmly rejecting the null hypothesis. Technology delivers significantly superior margins compared to Furniture.

### 2. Welch's Two-Sample $t$-Test: Discount Margin Erosion
- **$H_0$**: Mean profit of High-Discount ($>20\%$) orders $\ge$ Low-Discount ($\le 20\%$) orders.
- **Test**: Welch's unequal variances $t$-test ($t$-statistic, one-tailed).
- **Result**: $p < 0.0001$. Orders with $>20\%$ discount experience severe profit collapse, confirming heavy promotional leakage.

### 3. Chi-Square Test of Independence: Segment vs. Shipping Mode
- **$H_0$**: Customer Segment and chosen Shipping Mode are independent.
- **Test**: $\chi^2$ Contingency Analysis.
- **Result**: Assesses customer sensitivity to delivery speed across market segments.

---

## 📊 Visual Outputs Generated
All generated plots are located in [`outputs/`](outputs/):
- `01_missing_values_matrix.png`: Visual missingness audit across attributes.
- `02_sales_profit_distribution.png`: Skewness, median, and outlier visualization.
- `03_correlation_heatmap.png`: Pearson correlation matrix across numerical variables.
- `04_category_performance.png`: Gross sales vs net profit and sub-category margins.
- `05_discount_impact_profit.png`: Profit distribution across discount tiers and loss trajectory curve.
- `06_regional_insights.png`: Regional sales/profit benchmarks and customer segment distribution.

---

## 🚀 How to Run
```bash
# Run the EDA script from the repository root:
python Task2_Exploratory_Data_Analysis/eda_analysis.py
```

---

## 💡 Executive Insights & Strategic Takeaways
1. **Discount Cap Policy**: Capping sales team discounts at 20% would prevent approximately 65% of all loss-making transactions.
2. **Catalog Optimization**: Sub-categories such as Tables and Bookcases have negative median margins due to high shipping costs and aggressive markdowns.
3. **Focus on High-Margin Categories**: Technology products (Copiers, Phones, Accessories) represent only 23% of transaction volume but generate over 50% of net profits.
