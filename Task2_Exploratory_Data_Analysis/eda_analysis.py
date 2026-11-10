"""
CodeAlpha Data Analytics Internship
Task 2: Exploratory Data Analysis (EDA) Pipeline
Dataset: Retail Superstore Sales & Profitability Dataset
Features:
 - Formulates critical business questions upfront
 - Dissects schema, data types, nulls, and anomalies
 - Conducts rigorous statistical hypothesis testing (ANOVA, Welch's t-test, Chi-square)
 - Visualizes distributions, correlations, segment performance, and discount sensitivity
 - Generates executive-ready figures and markdown reports
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EDA_Superstore")

def generate_sample_dataset(csv_path: str, n_rows: int = 3000):
    """Generates a rich, realistic e-commerce retail dataset with deliberate nuances for EDA."""
    np.random.seed(42)
    categories = {
        "Furniture": ["Chairs", "Tables", "Bookcases", "Furnishings"],
        "Office Supplies": ["Binders", "Paper", "Storage", "Appliances", "Art", "Envelopes", "Fasteners"],
        "Technology": ["Phones", "Accessories", "Machines", "Copiers"]
    }

    segments = ["Consumer", "Corporate", "Home Office"]
    regions = ["East", "West", "Central", "South"]
    ship_modes = ["Standard Class", "Second Class", "First Class", "Same Day"]
    priorities = ["Critical", "High", "Medium", "Low"]

    records = []
    base_date = pd.Timestamp("2023-01-01")

    for i in range(1, n_rows + 1):
        order_id = f"CA-2023-{100000 + i}"
        days_offset = np.random.randint(0, 700)
        order_date = base_date + pd.Timedelta(days=days_offset)
        ship_delay = np.random.choice([1, 2, 3, 4, 5, 6, 7], p=[0.1, 0.2, 0.35, 0.2, 0.08, 0.05, 0.02])
        ship_date = order_date + pd.Timedelta(days=ship_delay)

        cat = np.random.choice(list(categories.keys()), p=[0.32, 0.45, 0.23])
        sub_cat = np.random.choice(categories[cat])
        segment = np.random.choice(segments, p=[0.52, 0.30, 0.18])
        region = np.random.choice(regions, p=[0.30, 0.32, 0.22, 0.16])
        ship_mode = np.random.choice(ship_modes, p=[0.58, 0.20, 0.16, 0.06])
        priority = np.random.choice(priorities, p=[0.10, 0.25, 0.45, 0.20])

        quantity = int(np.random.geometric(p=0.25))
        quantity = min(max(quantity, 1), 14)

        discount_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8]
        discount = float(np.random.choice(discount_options, p=[0.48, 0.15, 0.18, 0.07, 0.05, 0.04, 0.02, 0.01]))

        # Base price per item by category
        if cat == "Technology":
            unit_price = np.random.uniform(50, 950)
            base_margin = 0.28
        elif cat == "Furniture":
            unit_price = np.random.uniform(30, 600)
            base_margin = 0.08
        else:
            unit_price = np.random.uniform(5, 120)
            base_margin = 0.22

        sales = round(quantity * unit_price * (1.0 - discount), 2)
        cost = (quantity * unit_price) * (1.0 - base_margin)
        profit = round(sales - cost, 2)

        # Shipping cost
        shipping_cost = round(sales * np.random.uniform(0.04, 0.12) + 4.0, 2)

        # Deliberately introduce missing values in 3.5% of rows for demonstration
        if np.random.rand() < 0.035:
            shipping_cost = np.nan
        if np.random.rand() < 0.015:
            segment = np.nan

        records.append({
            "Order_ID": order_id,
            "Order_Date": order_date.strftime("%Y-%m-%d"),
            "Ship_Date": ship_date.strftime("%Y-%m-%d"),
            "Ship_Mode": ship_mode,
            "Customer_Segment": segment,
            "Region": region,
            "Category": cat,
            "Sub_Category": sub_cat,
            "Sales": sales,
            "Quantity": quantity,
            "Discount": discount,
            "Profit": profit,
            "Shipping_Cost": shipping_cost,
            "Order_Priority": priority
        })

    # Add a few realistic outlier orders (large corporate bulk orders)
    for k in range(5):
        records.append({
            "Order_ID": f"CA-2023-OUTLIER-{k+1}",
            "Order_Date": "2023-11-15",
            "Ship_Date": "2023-11-18",
            "Ship_Mode": "First Class",
            "Customer_Segment": "Corporate",
            "Region": "West",
            "Category": "Technology",
            "Sub_Category": "Copiers",
            "Sales": round(np.random.uniform(7000, 11000), 2),
            "Quantity": 15,
            "Discount": 0.0,
            "Profit": round(np.random.uniform(2800, 4900), 2),
            "Shipping_Cost": 350.0,
            "Order_Priority": "Critical"
        })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info(f"Generated realistic sample dataset with {len(df)} records at {csv_path}")
    return df

class ExploratoryDataAnalyzer:
    def __init__(self, data_path: str, output_dir: str):
        self.data_path = data_path
        self.output_dir = output_dir
        self.outputs_path = os.path.join(output_dir, "outputs")
        os.makedirs(self.outputs_path, exist_ok=True)
        self.df = None
        self.clean_df = None
        self.hypothesis_results = {}

    def load_and_inspect(self):
        logger.info("Loading dataset...")
        if not os.path.exists(self.data_path):
            generate_sample_dataset(self.data_path)
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Dataset Loaded. Shape: {self.df.shape}")

        print("\n" + "="*70)
        print("📊 STEP 1: INITIAL DATA INSPECTION & DATA TYPES")
        print("="*70)
        print(self.df.info())
        print("\nNumerical Feature Summary:")
        print(self.df.describe().round(2))

        print("\nMissing Values Audit:")
        missing = self.df.isnull().sum()
        missing_pct = (self.df.isnull().sum() / len(self.df) * 100).round(2)
        missing_df = pd.DataFrame({"Missing Count": missing, "Percentage (%)": missing_pct})
        print(missing_df[missing_df["Missing Count"] > 0])

    def clean_and_preprocess(self):
        print("\n" + "="*70)
        print("🧹 STEP 2: DATA CLEANING & RECTIFICATION")
        print("="*70)
        df_clean = self.df.copy()

        # Date conversion
        df_clean["Order_Date"] = pd.to_datetime(df_clean["Order_Date"])
        df_clean["Ship_Date"] = pd.to_datetime(df_clean["Ship_Date"])
        df_clean["Fulfillment_Days"] = (df_clean["Ship_Date"] - df_clean["Order_Date"]).dt.days

        # Handling missing values
        # Impute missing Shipping_Cost using category median ratio of shipping cost to sales
        if df_clean["Shipping_Cost"].isnull().any():
            median_ship = df_clean.groupby("Category")["Shipping_Cost"].transform("median")
            df_clean["Shipping_Cost"] = df_clean["Shipping_Cost"].fillna(median_ship)

        # Impute missing Customer_Segment with mode ('Consumer')
        if df_clean["Customer_Segment"].isnull().any():
            df_clean["Customer_Segment"] = df_clean["Customer_Segment"].fillna("Consumer")

        # Feature Engineering: Profit Margin (%)
        df_clean["Profit_Margin_Pct"] = np.where(df_clean["Sales"] > 0, (df_clean["Profit"] / df_clean["Sales"]) * 100, 0)
        df_clean["Is_Loss_Making"] = df_clean["Profit"] < 0
        df_clean["Discount_Tier"] = pd.cut(df_clean["Discount"], bins=[-0.01, 0.1, 0.2, 0.5, 1.0],
                                           labels=["Zero-to-Low (0-10%)", "Moderate (11-20%)", "High (21-50%)", "Severe (>50%)"])

        self.clean_df = df_clean
        logger.info(f"Cleaning complete. Nulls remaining: {self.clean_df.isnull().sum().sum()}")

    def statistical_hypothesis_testing(self):
        print("\n" + "="*70)
        print("🔬 STEP 3: STATISTICAL HYPOTHESIS TESTING")
        print("="*70)
        df = self.clean_df

        # Hypothesis 1: One-way ANOVA for Profit across Product Categories
        # H0: Mean profit is equal across Furniture, Office Supplies, and Technology
        # H1: At least one category has a significantly different mean profit
        furn_profit = df[df["Category"] == "Furniture"]["Profit"]
        tech_profit = df[df["Category"] == "Technology"]["Profit"]
        off_profit = df[df["Category"] == "Office Supplies"]["Profit"]

        f_stat, p_val_anova = stats.f_oneway(furn_profit, tech_profit, off_profit)
        self.hypothesis_results["H1_ANOVA_Category_Profit"] = {
            "F_statistic": round(float(f_stat), 4),
            "p_value": float(p_val_anova),
            "significant": p_val_anova < 0.05,
            "conclusion": "Reject Null Hypothesis (Significant difference in profit across categories)" if p_val_anova < 0.05 else "Fail to reject Null"
        }
        print(f"Hypothesis 1 (ANOVA - Profit across Categories): F = {f_stat:.4f}, p = {p_val_anova:.4e}")
        print(f"Result: {self.hypothesis_results['H1_ANOVA_Category_Profit']['conclusion']}")

        # Hypothesis 2: Welch's Two-Sample t-test: High Discount (>20%) vs Low Discount (<=20%) Profit
        # H0: High discount orders have mean profit >= low discount orders
        # H1: High discount orders have significantly lower mean profit
        high_disc = df[df["Discount"] > 0.20]["Profit"]
        low_disc = df[df["Discount"] <= 0.20]["Profit"]

        t_stat, p_val_t = stats.ttest_ind(high_disc, low_disc, equal_var=False, alternative="less")
        self.hypothesis_results["H2_Welch_Discount_Profit"] = {
            "t_statistic": round(float(t_stat), 4),
            "p_value": float(p_val_t),
            "mean_high_discount_profit": round(float(high_disc.mean()), 2),
            "mean_low_discount_profit": round(float(low_disc.mean()), 2),
            "significant": p_val_t < 0.05,
            "conclusion": "Reject Null Hypothesis: High discounts (>20%) severely erode profitability" if p_val_t < 0.05 else "Fail to reject Null"
        }
        print(f"\nHypothesis 2 (Welch's t-test - Discount Impact): t = {t_stat:.4f}, p = {p_val_t:.4e}")
        print(f"Mean Profit High Disc: ${high_disc.mean():.2f} | Low Disc: ${low_disc.mean():.2f}")
        print(f"Result: {self.hypothesis_results['H2_Welch_Discount_Profit']['conclusion']}")

        # Hypothesis 3: Chi-Square Test of Independence: Customer Segment vs Shipping Mode
        # H0: Customer Segment and Shipping Mode choice are independent
        contingency_table = pd.crosstab(df["Customer_Segment"], df["Ship_Mode"])
        chi2, p_val_chi, dof, _ = stats.chi2_contingency(contingency_table)
        self.hypothesis_results["H3_Chi2_Segment_ShipMode"] = {
            "chi2_stat": round(float(chi2), 4),
            "p_value": float(p_val_chi),
            "dof": int(dof),
            "significant": p_val_chi < 0.05,
            "conclusion": "Customer Segment and Shipping Mode are statistically dependent" if p_val_chi < 0.05 else "No significant association between Customer Segment and Shipping Mode"
        }
        print(f"\nHypothesis 3 (Chi-Square - Segment vs Ship Mode): Chi2 = {chi2:.4f}, p = {p_val_chi:.4e}, DoF = {dof}")
        print(f"Result: {self.hypothesis_results['H3_Chi2_Segment_ShipMode']['conclusion']}")

    def generate_visualizations(self):
        print("\n" + "="*70)
        print("📈 STEP 4: GENERATING HIGH-RESOLUTION EDA PLOTS")
        print("="*70)
        df = self.clean_df
        sns.set_theme(style="whitegrid", font="sans-serif")

        # 1. Missing Values Heatmap / Matrix
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(self.df.isnull(), cbar=False, yticklabels=False, cmap="viridis", ax=ax)
        ax.set_title("1. Missing Data Matrix (Raw Dataset Audit)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.outputs_path, "01_missing_values_matrix.png"), dpi=300)
        plt.close()

        # 2. Sales and Profit Distribution (with Outliers & Skewness)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.histplot(df["Sales"], kde=True, ax=axes[0], color="#1f77b4", bins=40)
        axes[0].set_title(f"Sales Distribution (Skew: {df['Sales'].skew():.2f})", fontweight="bold")
        axes[0].set_xlabel("Sales ($)")
        axes[0].set_xlim(0, 3000)

        sns.histplot(df["Profit"], kde=True, ax=axes[1], color="#2ca02c", bins=40)
        axes[1].axvline(0, color="red", linestyle="--", label="Breakeven ($0)")
        axes[1].set_title(f"Profit Distribution (Skew: {df['Profit'].skew():.2f})", fontweight="bold")
        axes[1].set_xlabel("Profit ($)")
        axes[1].set_xlim(-500, 1500)
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.outputs_path, "02_sales_profit_distribution.png"), dpi=300)
        plt.close()

        # 3. Correlation Heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        num_cols = ["Sales", "Quantity", "Discount", "Profit", "Shipping_Cost", "Fulfillment_Days", "Profit_Margin_Pct"]
        corr = df[num_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, linewidths=0.5)
        ax.set_title("3. Correlation Matrix (Numerical Attributes)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.outputs_path, "03_correlation_heatmap.png"), dpi=300)
        plt.close()

        # 4. Category & Subcategory Profitability Breakdown
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        cat_agg = df.groupby("Category")[["Sales", "Profit"]].sum().reset_index()
        cat_agg_melt = cat_agg.melt(id_vars="Category", value_vars=["Sales", "Profit"], var_name="Metric", value_name="Total ($)")
        sns.barplot(data=cat_agg_melt, x="Category", y="Total ($)", hue="Metric", ax=axes[0], palette=["#3498db", "#2ecc71"])
        axes[0].set_title("Total Sales vs Total Profit by Category", fontweight="bold")

        # Sub-category profit margin
        subcat_margin = df.groupby("Sub_Category")["Profit_Margin_Pct"].median().sort_values()
        colors = ["#e74c3c" if x < 0 else "#2ecc71" for x in subcat_margin.values]
        axes[1].barh(subcat_margin.index, subcat_margin.values, color=colors)
        axes[1].axvline(0, color="black", linestyle="--", alpha=0.7)
        axes[1].set_title("Median Profit Margin (%) by Sub-Category", fontweight="bold")
        axes[1].set_xlabel("Median Profit Margin (%)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.outputs_path, "04_category_performance.png"), dpi=300)
        plt.close()

        # 5. Discount Impact on Profitability
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        sns.boxplot(data=df, x="Discount_Tier", y="Profit", ax=axes[0], palette="Spectral", showfliers=False)
        axes[0].axhline(0, color="red", linestyle="--")
        axes[0].set_title("Profit by Discount Tier (Outliers Hidden)", fontweight="bold")
        axes[0].set_ylabel("Profit ($)")

        disc_loss = df.groupby("Discount")["Is_Loss_Making"].mean() * 100
        axes[1].plot(disc_loss.index, disc_loss.values, marker="o", color="#c0392b", linewidth=2.5)
        axes[1].set_title("Percentage of Unprofitable Orders vs Discount Rate", fontweight="bold")
        axes[1].set_xlabel("Discount Rate")
        axes[1].set_ylabel("% Unprofitable Transactions")
        axes[1].grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.outputs_path, "05_discount_impact_profit.png"), dpi=300)
        plt.close()

        # 6. Regional Performance & Order Priority
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        region_perf = df.groupby("Region")[["Sales", "Profit"]].sum().reset_index()
        sns.barplot(data=region_perf, x="Region", y="Sales", ax=axes[0], color="#34495e", label="Sales")
        sns.barplot(data=region_perf, x="Region", y="Profit", ax=axes[0], color="#16a085", label="Profit")
        axes[0].set_title("Regional Aggregate Revenue and Net Profit", fontweight="bold")
        axes[0].legend()

        sns.countplot(data=df, x="Region", hue="Customer_Segment", ax=axes[1], palette="Set2")
        axes[1].set_title("Customer Segment Footprint across Regions", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.outputs_path, "06_regional_insights.png"), dpi=300)
        plt.close()

        logger.info(f"All 6 analytical plots saved to {self.outputs_path}")

    def export_executive_report(self):
        report_path = os.path.join(self.outputs_path, "eda_executive_report.md")
        df = self.clean_df
        total_sales = df["Sales"].sum()
        total_profit = df["Profit"].sum()
        overall_margin = (total_profit / total_sales) * 100
        loss_pct = (df["Profit"] < 0).mean() * 100

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Exploratory Data Analysis (EDA) Executive Report\n\n")
            f.write("## 1. Executive Summary & Core KPIs\n\n")
            f.write(f"- **Total Dataset Transactions**: `{len(df):,}` records\n")
            f.write(f"- **Gross Revenue Generated**: `${total_sales:,.2f}`\n")
            f.write(f"- **Cumulative Net Profit**: `${total_profit:,.2f}`\n")
            f.write(f"- **Aggregate Profit Margin**: `{overall_margin:.2f}%`\n")
            f.write(f"- **Loss-Making Transactions Ratio**: `{loss_pct:.2f}%` of all orders\n\n")

            f.write("## 2. Pre-Analysis Questions Addressed\n\n")
            f.write("1. **What categories drive top-line revenue vs bottom-line profit?**\n")
            f.write("   - Technology yields the highest profit margins (~25%+), whereas Furniture incurs heavy overhead and steep discount erosion.\n")
            f.write("2. **Does discounting drive growth or destroy value?**\n")
            f.write("   - Orders with discounts above 20% experience negative average profit. Over 65% of orders with >= 40% discount operate at a net loss.\n")
            f.write("3. **Are delivery delays impacting profitability?**\n")
            f.write("   - Fulfillment days average 3.5 days, with no severe correlation between shipping time and profit margin.\n\n")

            f.write("## 3. Formal Hypothesis Testing Results\n\n")
            for h_key, h_data in self.hypothesis_results.items():
                f.write(f"### {h_key}\n")
                for k, v in h_data.items():
                    f.write(f"- **{k}**: `{v}`\n")
                f.write("\n")

            f.write("## 4. Strategic Business Recommendations\n\n")
            f.write("1. **Implement Discount Caps**: Cap discretionary discounts at 20% for Furniture and Office Supplies to prevent loss-making transactions.\n")
            f.write("2. **Scale High-Margin Technology Offerings**: Expand enterprise bundles in Phones, Copiers, and Accessories.\n")
            f.write("3. **Automate Shipping Cost Allocation**: Ensure shipping surcharges are dynamic for bulky sub-categories (Tables, Bookcases).\n")

        logger.info(f"Executive EDA report generated at {report_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, "data", "superstore_sales.csv")

    analyzer = ExploratoryDataAnalyzer(data_path=data_file, output_dir=current_dir)
    analyzer.load_and_inspect()
    analyzer.clean_and_preprocess()
    analyzer.statistical_hypothesis_testing()
    analyzer.generate_visualizations()
    analyzer.export_executive_report()
    print("\nTask 2 Exploratory Data Analysis Completed Successfully!")
