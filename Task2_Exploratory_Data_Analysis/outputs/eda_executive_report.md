# Exploratory Data Analysis (EDA) Executive Report

## 1. Executive Summary & Core KPIs

- **Total Dataset Transactions**: `3,005` records
- **Gross Revenue Generated**: `$2,654,570.25`
- **Cumulative Net Profit**: `$202,189.29`
- **Aggregate Profit Margin**: `7.62%`
- **Loss-Making Transactions Ratio**: `29.82%` of all orders

## 2. Pre-Analysis Questions Addressed

1. **What categories drive top-line revenue vs bottom-line profit?**
   - Technology yields the highest profit margins (~25%+), whereas Furniture incurs heavy overhead and steep discount erosion.
2. **Does discounting drive growth or destroy value?**
   - Orders with discounts above 20% experience negative average profit. Over 65% of orders with >= 40% discount operate at a net loss.
3. **Are delivery delays impacting profitability?**
   - Fulfillment days average 3.5 days, with no severe correlation between shipping time and profit margin.

## 3. Formal Hypothesis Testing Results

### H1_ANOVA_Category_Profit
- **F_statistic**: `219.5498`
- **p_value**: `1.0253691256813718e-89`
- **significant**: `True`
- **conclusion**: `Reject Null Hypothesis (Significant difference in profit across categories)`

### H2_Welch_Discount_Profit
- **t_statistic**: `-19.4385`
- **p_value**: `9.384366050372088e-70`
- **mean_high_discount_profit**: `-230.98`
- **mean_low_discount_profit**: `138.62`
- **significant**: `True`
- **conclusion**: `Reject Null Hypothesis: High discounts (>20%) severely erode profitability`

### H3_Chi2_Segment_ShipMode
- **chi2_stat**: `3.9276`
- **p_value**: `0.6864695515009394`
- **dof**: `6`
- **significant**: `False`
- **conclusion**: `No significant association between Customer Segment and Shipping Mode`

## 4. Strategic Business Recommendations

1. **Implement Discount Caps**: Cap discretionary discounts at 20% for Furniture and Office Supplies to prevent loss-making transactions.
2. **Scale High-Margin Technology Offerings**: Expand enterprise bundles in Phones, Copiers, and Accessories.
3. **Automate Shipping Cost Allocation**: Ensure shipping surcharges are dynamic for bulky sub-categories (Tables, Bookcases).
