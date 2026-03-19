"""
Instruction Dataset Creator for Text-to-SQL Fine-Tuning

Generates an instruction-tuning dataset from:
1. Golden queries (data/golden_queries.json) - deduplicated
2. Programmatically generated variations covering Olist schema patterns

Output format (Alpaca-style):
{
    "instruction": "system prompt with schema context",
    "input": "natural language question",
    "output": "SQL query"
}

Usage:
    python scripts/create_instruction_dataset.py
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

# Olist schema definition used in instruction prompts
OLIST_SCHEMA = """Database: ANALYTICS_COPILOT (Snowflake)
Schema: RAW

Tables and columns:
- ORDERS: ORDER_ID, CUSTOMER_ID, ORDER_STATUS, ORDER_PURCHASE_TIMESTAMP, ORDER_APPROVED_AT, ORDER_DELIVERED_CARRIER_DATE, ORDER_DELIVERED_CUSTOMER_DATE, ORDER_ESTIMATED_DELIVERY_DATE
- CUSTOMERS: CUSTOMER_ID, CUSTOMER_UNIQUE_ID, CUSTOMER_ZIP_CODE_PREFIX, CUSTOMER_CITY, CUSTOMER_STATE
- ORDER_ITEMS: ORDER_ID, ORDER_ITEM_ID, PRODUCT_ID, SELLER_ID, SHIPPING_LIMIT_DATE, PRICE, FREIGHT_VALUE
- ORDER_PAYMENTS: ORDER_ID, PAYMENT_SEQUENTIAL, PAYMENT_TYPE, PAYMENT_INSTALLMENTS, PAYMENT_VALUE
- ORDER_REVIEWS: REVIEW_ID, ORDER_ID, REVIEW_SCORE, REVIEW_COMMENT_TITLE, REVIEW_COMMENT_MESSAGE, REVIEW_CREATION_DATE, REVIEW_ANSWER_TIMESTAMP
- PRODUCTS: PRODUCT_ID, PRODUCT_CATEGORY_NAME, PRODUCT_NAME_LENGTH, PRODUCT_DESCRIPTION_LENGTH, PRODUCT_PHOTOS_QTY, PRODUCT_WEIGHT_G, PRODUCT_LENGTH_CM, PRODUCT_HEIGHT_CM, PRODUCT_WIDTH_CM
- SELLERS: SELLER_ID, SELLER_ZIP_CODE_PREFIX, SELLER_CITY, SELLER_STATE
- GEOLOCATION: GEOLOCATION_ZIP_CODE_PREFIX, GEOLOCATION_LAT, GEOLOCATION_LNG, GEOLOCATION_CITY, GEOLOCATION_STATE
- PRODUCT_CATEGORY_TRANSLATION: PRODUCT_CATEGORY_NAME, PRODUCT_CATEGORY_NAME_ENGLISH

Key relationships:
- ORDERS.CUSTOMER_ID -> CUSTOMERS.CUSTOMER_ID
- ORDER_ITEMS.ORDER_ID -> ORDERS.ORDER_ID
- ORDER_ITEMS.PRODUCT_ID -> PRODUCTS.PRODUCT_ID
- ORDER_ITEMS.SELLER_ID -> SELLERS.SELLER_ID
- ORDER_REVIEWS.ORDER_ID -> ORDERS.ORDER_ID
- ORDER_PAYMENTS.ORDER_ID -> ORDERS.ORDER_ID
- PRODUCTS.PRODUCT_CATEGORY_NAME -> PRODUCT_CATEGORY_TRANSLATION.PRODUCT_CATEGORY_NAME

Use fully qualified table names: ANALYTICS_COPILOT.RAW.<TABLE>
Use Snowflake SQL syntax."""

SYSTEM_INSTRUCTION = f"""You are a Snowflake SQL expert. Given a natural language question about the Olist Brazilian E-Commerce dataset, generate a correct Snowflake SQL query.

{OLIST_SCHEMA}

Return ONLY the SQL query, no explanations."""


def load_golden_queries() -> list[dict]:
    """Load and deduplicate golden queries."""
    path = Path(__file__).parent.parent / "data" / "golden_queries.json"
    with open(path) as f:
        queries = json.load(f)

    # Deduplicate by (question, sql_query) pair
    seen = set()
    unique = []
    for q in queries:
        key = (q["question"], q["sql_query"])
        if key not in seen:
            seen.add(key)
            unique.append(q)
    return unique


def qualify_sql(sql: str) -> str:
    """Add ANALYTICS_COPILOT.RAW. prefix to table references in SQL."""
    tables = [
        "ORDERS", "CUSTOMERS", "ORDER_ITEMS", "ORDER_PAYMENTS",
        "ORDER_REVIEWS", "PRODUCTS", "SELLERS", "GEOLOCATION",
        "PRODUCT_CATEGORY_TRANSLATION", "SUPERSTORE_SALES"
    ]
    result = sql
    for table in tables:
        # Replace RAW.TABLE with fully qualified name
        result = result.replace(f"RAW.{table}", f"ANALYTICS_COPILOT.RAW.{table}")
        # Replace standalone FROM/JOIN TABLE (not already qualified)
        for keyword in ["FROM ", "JOIN "]:
            result = result.replace(f"{keyword}{table} ", f"{keyword}ANALYTICS_COPILOT.RAW.{table} ")
            result = result.replace(f"{keyword}{table}\n", f"{keyword}ANALYTICS_COPILOT.RAW.{table}\n")
    return result


def generate_augmented_examples() -> list[dict]:
    """Generate additional training examples covering diverse SQL patterns."""
    examples = [
        # Simple aggregations
        {
            "input": "How many orders are there in total?",
            "output": "SELECT COUNT(*) AS total_orders FROM ANALYTICS_COPILOT.RAW.ORDERS"
        },
        {
            "input": "What is the total revenue from all order items?",
            "output": "SELECT SUM(PRICE) AS total_revenue FROM ANALYTICS_COPILOT.RAW.ORDER_ITEMS"
        },
        {
            "input": "What is the average freight value per order item?",
            "output": "SELECT AVG(FREIGHT_VALUE) AS avg_freight FROM ANALYTICS_COPILOT.RAW.ORDER_ITEMS"
        },
        {
            "input": "How many unique customers have placed orders?",
            "output": "SELECT COUNT(DISTINCT CUSTOMER_UNIQUE_ID) AS unique_customers FROM ANALYTICS_COPILOT.RAW.CUSTOMERS"
        },
        {
            "input": "What is the total payment value across all orders?",
            "output": "SELECT SUM(PAYMENT_VALUE) AS total_payments FROM ANALYTICS_COPILOT.RAW.ORDER_PAYMENTS"
        },
        # Filtering
        {
            "input": "How many orders have status 'delivered'?",
            "output": "SELECT COUNT(*) AS delivered_orders FROM ANALYTICS_COPILOT.RAW.ORDERS WHERE ORDER_STATUS = 'delivered'"
        },
        {
            "input": "What is the total revenue from credit card payments?",
            "output": "SELECT SUM(PAYMENT_VALUE) AS total_cc_payments FROM ANALYTICS_COPILOT.RAW.ORDER_PAYMENTS WHERE PAYMENT_TYPE = 'credit_card'"
        },
        {
            "input": "How many products weigh more than 5000 grams?",
            "output": "SELECT COUNT(*) AS heavy_products FROM ANALYTICS_COPILOT.RAW.PRODUCTS WHERE PRODUCT_WEIGHT_G > 5000"
        },
        {
            "input": "How many orders were placed between January and June 2018?",
            "output": "SELECT COUNT(*) AS order_count FROM ANALYTICS_COPILOT.RAW.ORDERS WHERE ORDER_PURCHASE_TIMESTAMP >= '2018-01-01' AND ORDER_PURCHASE_TIMESTAMP < '2018-07-01'"
        },
        {
            "input": "How many customers are from the state of Bahia?",
            "output": "SELECT COUNT(*) AS customer_count FROM ANALYTICS_COPILOT.RAW.CUSTOMERS WHERE CUSTOMER_STATE = 'BA'"
        },
        # GROUP BY
        {
            "input": "What is the total revenue by payment type?",
            "output": "SELECT PAYMENT_TYPE, SUM(PAYMENT_VALUE) AS total_revenue FROM ANALYTICS_COPILOT.RAW.ORDER_PAYMENTS GROUP BY PAYMENT_TYPE ORDER BY total_revenue DESC"
        },
        {
            "input": "How many orders per month in 2018?",
            "output": "SELECT DATE_TRUNC('MONTH', ORDER_PURCHASE_TIMESTAMP) AS order_month, COUNT(*) AS order_count FROM ANALYTICS_COPILOT.RAW.ORDERS WHERE EXTRACT(YEAR FROM ORDER_PURCHASE_TIMESTAMP) = 2018 GROUP BY DATE_TRUNC('MONTH', ORDER_PURCHASE_TIMESTAMP) ORDER BY order_month"
        },
        {
            "input": "What is the average review score by order status?",
            "output": "SELECT o.ORDER_STATUS, AVG(r.REVIEW_SCORE) AS avg_review FROM ANALYTICS_COPILOT.RAW.ORDERS o JOIN ANALYTICS_COPILOT.RAW.ORDER_REVIEWS r ON o.ORDER_ID = r.ORDER_ID GROUP BY o.ORDER_STATUS ORDER BY avg_review DESC"
        },
        {
            "input": "How many sellers per state?",
            "output": "SELECT SELLER_STATE, COUNT(*) AS seller_count FROM ANALYTICS_COPILOT.RAW.SELLERS GROUP BY SELLER_STATE ORDER BY seller_count DESC"
        },
        {
            "input": "What is the average product weight by category?",
            "output": "SELECT PRODUCT_CATEGORY_NAME, AVG(PRODUCT_WEIGHT_G) AS avg_weight FROM ANALYTICS_COPILOT.RAW.PRODUCTS GROUP BY PRODUCT_CATEGORY_NAME ORDER BY avg_weight DESC"
        },
        # JOINs
        {
            "input": "What is the total revenue by customer state?",
            "output": "SELECT c.CUSTOMER_STATE, SUM(oi.PRICE) AS total_revenue FROM ANALYTICS_COPILOT.RAW.ORDER_ITEMS oi JOIN ANALYTICS_COPILOT.RAW.ORDERS o ON oi.ORDER_ID = o.ORDER_ID JOIN ANALYTICS_COPILOT.RAW.CUSTOMERS c ON o.CUSTOMER_ID = c.CUSTOMER_ID GROUP BY c.CUSTOMER_STATE ORDER BY total_revenue DESC"
        },
        {
            "input": "What is the average delivery time in days by customer state?",
            "output": "SELECT c.CUSTOMER_STATE, AVG(DATEDIFF('day', o.ORDER_PURCHASE_TIMESTAMP, o.ORDER_DELIVERED_CUSTOMER_DATE)) AS avg_delivery_days FROM ANALYTICS_COPILOT.RAW.ORDERS o JOIN ANALYTICS_COPILOT.RAW.CUSTOMERS c ON o.CUSTOMER_ID = c.CUSTOMER_ID WHERE o.ORDER_DELIVERED_CUSTOMER_DATE IS NOT NULL GROUP BY c.CUSTOMER_STATE ORDER BY avg_delivery_days"
        },
        {
            "input": "What are the top 10 product categories by total sales?",
            "output": "SELECT p.PRODUCT_CATEGORY_NAME, SUM(oi.PRICE) AS total_sales FROM ANALYTICS_COPILOT.RAW.ORDER_ITEMS oi JOIN ANALYTICS_COPILOT.RAW.PRODUCTS p ON oi.PRODUCT_ID = p.PRODUCT_ID GROUP BY p.PRODUCT_CATEGORY_NAME ORDER BY total_sales DESC LIMIT 10"
        },
        {
            "input": "What is the average review score for each product category?",
            "output": "SELECT p.PRODUCT_CATEGORY_NAME, AVG(r.REVIEW_SCORE) AS avg_score FROM ANALYTICS_COPILOT.RAW.PRODUCTS p JOIN ANALYTICS_COPILOT.RAW.ORDER_ITEMS oi ON p.PRODUCT_ID = oi.PRODUCT_ID JOIN ANALYTICS_COPILOT.RAW.ORDER_REVIEWS r ON oi.ORDER_ID = r.ORDER_ID GROUP BY p.PRODUCT_CATEGORY_NAME ORDER BY avg_score DESC"
        },
        {
            "input": "Show me the English category names with total revenue",
            "output": "SELECT t.PRODUCT_CATEGORY_NAME_ENGLISH, SUM(oi.PRICE) AS total_revenue FROM ANALYTICS_COPILOT.RAW.ORDER_ITEMS oi JOIN ANALYTICS_COPILOT.RAW.PRODUCTS p ON oi.PRODUCT_ID = p.PRODUCT_ID JOIN ANALYTICS_COPILOT.RAW.PRODUCT_CATEGORY_TRANSLATION t ON p.PRODUCT_CATEGORY_NAME = t.PRODUCT_CATEGORY_NAME GROUP BY t.PRODUCT_CATEGORY_NAME_ENGLISH ORDER BY total_revenue DESC"
        },
        # Date functions
        {
            "input": "What is the average delivery time in days?",
            "output": "SELECT AVG(DATEDIFF('day', ORDER_PURCHASE_TIMESTAMP, ORDER_DELIVERED_CUSTOMER_DATE)) AS avg_delivery_days FROM ANALYTICS_COPILOT.RAW.ORDERS WHERE ORDER_DELIVERED_CUSTOMER_DATE IS NOT NULL"
        },
        {
            "input": "How many orders were placed each quarter in 2018?",
            "output": "SELECT DATE_TRUNC('QUARTER', ORDER_PURCHASE_TIMESTAMP) AS quarter, COUNT(*) AS order_count FROM ANALYTICS_COPILOT.RAW.ORDERS WHERE EXTRACT(YEAR FROM ORDER_PURCHASE_TIMESTAMP) = 2018 GROUP BY DATE_TRUNC('QUARTER', ORDER_PURCHASE_TIMESTAMP) ORDER BY quarter"
        },
        {
            "input": "What percentage of orders were delivered late?",
            "output": "SELECT COUNT(CASE WHEN ORDER_DELIVERED_CUSTOMER_DATE > ORDER_ESTIMATED_DELIVERY_DATE THEN 1 END) * 100.0 / COUNT(*) AS late_delivery_pct FROM ANALYTICS_COPILOT.RAW.ORDERS WHERE ORDER_DELIVERED_CUSTOMER_DATE IS NOT NULL AND ORDER_ESTIMATED_DELIVERY_DATE IS NOT NULL"
        },
        # Window functions / CTEs
        {
            "input": "Rank customer states by total number of orders",
            "output": "SELECT c.CUSTOMER_STATE, COUNT(o.ORDER_ID) AS order_count, RANK() OVER (ORDER BY COUNT(o.ORDER_ID) DESC) AS state_rank FROM ANALYTICS_COPILOT.RAW.ORDERS o JOIN ANALYTICS_COPILOT.RAW.CUSTOMERS c ON o.CUSTOMER_ID = c.CUSTOMER_ID GROUP BY c.CUSTOMER_STATE ORDER BY state_rank"
        },
        {
            "input": "What is the running total of monthly revenue?",
            "output": "WITH monthly_rev AS (SELECT DATE_TRUNC('MONTH', o.ORDER_PURCHASE_TIMESTAMP) AS month, SUM(oi.PRICE) AS revenue FROM ANALYTICS_COPILOT.RAW.ORDER_ITEMS oi JOIN ANALYTICS_COPILOT.RAW.ORDERS o ON oi.ORDER_ID = o.ORDER_ID GROUP BY DATE_TRUNC('MONTH', o.ORDER_PURCHASE_TIMESTAMP)) SELECT month, revenue, SUM(revenue) OVER (ORDER BY month) AS running_total FROM monthly_rev ORDER BY month"
        },
        {
            "input": "Find the top 2 sellers by revenue in each state",
            "output": "WITH ranked AS (SELECT s.SELLER_STATE, s.SELLER_ID, SUM(oi.PRICE) AS total_revenue, ROW_NUMBER() OVER (PARTITION BY s.SELLER_STATE ORDER BY SUM(oi.PRICE) DESC) AS rn FROM ANALYTICS_COPILOT.RAW.ORDER_ITEMS oi JOIN ANALYTICS_COPILOT.RAW.SELLERS s ON oi.SELLER_ID = s.SELLER_ID GROUP BY s.SELLER_STATE, s.SELLER_ID) SELECT SELLER_STATE, SELLER_ID, total_revenue FROM ranked WHERE rn <= 2 ORDER BY SELLER_STATE, total_revenue DESC"
        },
        # Subqueries and HAVING
        {
            "input": "Which product categories have more than 1000 orders?",
            "output": "SELECT p.PRODUCT_CATEGORY_NAME, COUNT(*) AS order_count FROM ANALYTICS_COPILOT.RAW.ORDER_ITEMS oi JOIN ANALYTICS_COPILOT.RAW.PRODUCTS p ON oi.PRODUCT_ID = p.PRODUCT_ID GROUP BY p.PRODUCT_CATEGORY_NAME HAVING COUNT(*) > 1000 ORDER BY order_count DESC"
        },
        {
            "input": "Which sellers have an average review score below 3?",
            "output": "SELECT oi.SELLER_ID, AVG(r.REVIEW_SCORE) AS avg_score FROM ANALYTICS_COPILOT.RAW.ORDER_ITEMS oi JOIN ANALYTICS_COPILOT.RAW.ORDER_REVIEWS r ON oi.ORDER_ID = r.ORDER_ID GROUP BY oi.SELLER_ID HAVING AVG(r.REVIEW_SCORE) < 3 ORDER BY avg_score"
        },
        {
            "input": "What are the top 5 cities with the most sellers?",
            "output": "SELECT SELLER_CITY, COUNT(*) AS seller_count FROM ANALYTICS_COPILOT.RAW.SELLERS GROUP BY SELLER_CITY ORDER BY seller_count DESC LIMIT 5"
        },
        {
            "input": "Which states have average order value above 200?",
            "output": "SELECT c.CUSTOMER_STATE, AVG(oi.PRICE + oi.FREIGHT_VALUE) AS avg_order_value FROM ANALYTICS_COPILOT.RAW.ORDER_ITEMS oi JOIN ANALYTICS_COPILOT.RAW.ORDERS o ON oi.ORDER_ID = o.ORDER_ID JOIN ANALYTICS_COPILOT.RAW.CUSTOMERS c ON o.CUSTOMER_ID = c.CUSTOMER_ID GROUP BY c.CUSTOMER_STATE HAVING AVG(oi.PRICE + oi.FREIGHT_VALUE) > 200 ORDER BY avg_order_value DESC"
        },
        {
            "input": "How many orders have more than 3 items?",
            "output": "SELECT COUNT(*) AS multi_item_orders FROM (SELECT ORDER_ID, COUNT(*) AS item_count FROM ANALYTICS_COPILOT.RAW.ORDER_ITEMS GROUP BY ORDER_ID HAVING COUNT(*) > 3)"
        },
        # Payment analysis
        {
            "input": "What is the average number of payment installments by payment type?",
            "output": "SELECT PAYMENT_TYPE, AVG(PAYMENT_INSTALLMENTS) AS avg_installments FROM ANALYTICS_COPILOT.RAW.ORDER_PAYMENTS GROUP BY PAYMENT_TYPE ORDER BY avg_installments DESC"
        },
        {
            "input": "What percentage of payments are made by credit card?",
            "output": "SELECT COUNT(CASE WHEN PAYMENT_TYPE = 'credit_card' THEN 1 END) * 100.0 / COUNT(*) AS cc_pct FROM ANALYTICS_COPILOT.RAW.ORDER_PAYMENTS"
        },
        # Seller analysis
        {
            "input": "What is the average number of items sold per seller?",
            "output": "SELECT AVG(item_count) AS avg_items_per_seller FROM (SELECT SELLER_ID, COUNT(*) AS item_count FROM ANALYTICS_COPILOT.RAW.ORDER_ITEMS GROUP BY SELLER_ID)"
        },
        {
            "input": "Which seller has the highest total revenue?",
            "output": "SELECT SELLER_ID, SUM(PRICE) AS total_revenue FROM ANALYTICS_COPILOT.RAW.ORDER_ITEMS GROUP BY SELLER_ID ORDER BY total_revenue DESC LIMIT 1"
        },
        # Product analysis
        {
            "input": "What is the correlation between product weight and freight value?",
            "output": "SELECT CORR(p.PRODUCT_WEIGHT_G, oi.FREIGHT_VALUE) AS weight_freight_correlation FROM ANALYTICS_COPILOT.RAW.ORDER_ITEMS oi JOIN ANALYTICS_COPILOT.RAW.PRODUCTS p ON oi.PRODUCT_ID = p.PRODUCT_ID"
        },
        {
            "input": "What is the average product description length by category?",
            "output": "SELECT PRODUCT_CATEGORY_NAME, AVG(PRODUCT_DESCRIPTION_LENGTH) AS avg_desc_length FROM ANALYTICS_COPILOT.RAW.PRODUCTS GROUP BY PRODUCT_CATEGORY_NAME ORDER BY avg_desc_length DESC"
        },
        {
            "input": "How many products have more than 3 photos?",
            "output": "SELECT COUNT(*) AS products_with_many_photos FROM ANALYTICS_COPILOT.RAW.PRODUCTS WHERE PRODUCT_PHOTOS_QTY > 3"
        },
        # Complex multi-join
        {
            "input": "What is the average review score by seller state?",
            "output": "SELECT s.SELLER_STATE, AVG(r.REVIEW_SCORE) AS avg_score FROM ANALYTICS_COPILOT.RAW.SELLERS s JOIN ANALYTICS_COPILOT.RAW.ORDER_ITEMS oi ON s.SELLER_ID = oi.SELLER_ID JOIN ANALYTICS_COPILOT.RAW.ORDER_REVIEWS r ON oi.ORDER_ID = r.ORDER_ID GROUP BY s.SELLER_STATE ORDER BY avg_score DESC"
        },
        {
            "input": "What is the total revenue by English product category name?",
            "output": "SELECT t.PRODUCT_CATEGORY_NAME_ENGLISH, SUM(oi.PRICE) AS total_revenue FROM ANALYTICS_COPILOT.RAW.ORDER_ITEMS oi JOIN ANALYTICS_COPILOT.RAW.PRODUCTS p ON oi.PRODUCT_ID = p.PRODUCT_ID JOIN ANALYTICS_COPILOT.RAW.PRODUCT_CATEGORY_TRANSLATION t ON p.PRODUCT_CATEGORY_NAME = t.PRODUCT_CATEGORY_NAME GROUP BY t.PRODUCT_CATEGORY_NAME_ENGLISH ORDER BY total_revenue DESC"
        },
        {
            "input": "Show monthly order count and average review score over time",
            "output": "SELECT DATE_TRUNC('MONTH', o.ORDER_PURCHASE_TIMESTAMP) AS month, COUNT(DISTINCT o.ORDER_ID) AS order_count, AVG(r.REVIEW_SCORE) AS avg_review FROM ANALYTICS_COPILOT.RAW.ORDERS o JOIN ANALYTICS_COPILOT.RAW.ORDER_REVIEWS r ON o.ORDER_ID = r.ORDER_ID GROUP BY DATE_TRUNC('MONTH', o.ORDER_PURCHASE_TIMESTAMP) ORDER BY month"
        },
        {
            "input": "What is the average time between order approval and delivery by state?",
            "output": "SELECT c.CUSTOMER_STATE, AVG(DATEDIFF('day', o.ORDER_APPROVED_AT, o.ORDER_DELIVERED_CUSTOMER_DATE)) AS avg_approval_to_delivery_days FROM ANALYTICS_COPILOT.RAW.ORDERS o JOIN ANALYTICS_COPILOT.RAW.CUSTOMERS c ON o.CUSTOMER_ID = c.CUSTOMER_ID WHERE o.ORDER_APPROVED_AT IS NOT NULL AND o.ORDER_DELIVERED_CUSTOMER_DATE IS NOT NULL GROUP BY c.CUSTOMER_STATE ORDER BY avg_approval_to_delivery_days"
        },
        # More variations for robustness
        {
            "input": "List all distinct order statuses",
            "output": "SELECT DISTINCT ORDER_STATUS FROM ANALYTICS_COPILOT.RAW.ORDERS ORDER BY ORDER_STATUS"
        },
        {
            "input": "What are the distinct payment types?",
            "output": "SELECT DISTINCT PAYMENT_TYPE FROM ANALYTICS_COPILOT.RAW.ORDER_PAYMENTS ORDER BY PAYMENT_TYPE"
        },
        {
            "input": "Count orders by status",
            "output": "SELECT ORDER_STATUS, COUNT(*) AS order_count FROM ANALYTICS_COPILOT.RAW.ORDERS GROUP BY ORDER_STATUS ORDER BY order_count DESC"
        },
        {
            "input": "What is the median payment value?",
            "output": "SELECT MEDIAN(PAYMENT_VALUE) AS median_payment FROM ANALYTICS_COPILOT.RAW.ORDER_PAYMENTS"
        },
        {
            "input": "Show the number of orders and average payment per customer state",
            "output": "SELECT c.CUSTOMER_STATE, COUNT(DISTINCT o.ORDER_ID) AS order_count, AVG(p.PAYMENT_VALUE) AS avg_payment FROM ANALYTICS_COPILOT.RAW.ORDERS o JOIN ANALYTICS_COPILOT.RAW.CUSTOMERS c ON o.CUSTOMER_ID = c.CUSTOMER_ID JOIN ANALYTICS_COPILOT.RAW.ORDER_PAYMENTS p ON o.ORDER_ID = p.ORDER_ID GROUP BY c.CUSTOMER_STATE ORDER BY order_count DESC"
        },
        {
            "input": "What is the month with the highest number of orders?",
            "output": "SELECT DATE_TRUNC('MONTH', ORDER_PURCHASE_TIMESTAMP) AS month, COUNT(*) AS order_count FROM ANALYTICS_COPILOT.RAW.ORDERS GROUP BY DATE_TRUNC('MONTH', ORDER_PURCHASE_TIMESTAMP) ORDER BY order_count DESC LIMIT 1"
        },
        {
            "input": "How many orders were cancelled?",
            "output": "SELECT COUNT(*) AS cancelled_orders FROM ANALYTICS_COPILOT.RAW.ORDERS WHERE ORDER_STATUS = 'canceled'"
        },
        {
            "input": "What is the total and average freight by seller state?",
            "output": "SELECT s.SELLER_STATE, SUM(oi.FREIGHT_VALUE) AS total_freight, AVG(oi.FREIGHT_VALUE) AS avg_freight FROM ANALYTICS_COPILOT.RAW.ORDER_ITEMS oi JOIN ANALYTICS_COPILOT.RAW.SELLERS s ON oi.SELLER_ID = s.SELLER_ID GROUP BY s.SELLER_STATE ORDER BY total_freight DESC"
        },
    ]
    return examples


def build_dataset():
    """Build the complete instruction dataset."""
    dataset = []

    # 1. Convert golden queries
    golden = load_golden_queries()
    print(f"Loaded {len(golden)} unique golden queries")

    for q in golden:
        sql = qualify_sql(q["sql_query"])
        dataset.append({
            "instruction": SYSTEM_INSTRUCTION,
            "input": q["question"],
            "output": sql,
            "difficulty": q["difficulty"],
            "source": "golden"
        })

    # 2. Add augmented examples
    augmented = generate_augmented_examples()
    print(f"Generated {len(augmented)} augmented examples")

    for ex in augmented:
        dataset.append({
            "instruction": SYSTEM_INSTRUCTION,
            "input": ex["input"],
            "output": ex["output"],
            "difficulty": "augmented",
            "source": "augmented"
        })

    print(f"Total dataset size: {len(dataset)} examples")
    return dataset


def main():
    dataset = build_dataset()

    # Save full dataset
    output_path = Path(__file__).parent.parent / "data" / "instruction_dataset.json"
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved instruction dataset to {output_path}")

    # Save train/val split (90/10)
    import random
    random.seed(42)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split = int(len(indices) * 0.9)
    train = [dataset[i] for i in indices[:split]]
    val = [dataset[i] for i in indices[split:]]

    train_path = Path(__file__).parent.parent / "data" / "instruction_train.json"
    val_path = Path(__file__).parent.parent / "data" / "instruction_val.json"

    with open(train_path, "w") as f:
        json.dump(train, f, indent=2)
    with open(val_path, "w") as f:
        json.dump(val, f, indent=2)

    print(f"Train: {len(train)} examples -> {train_path}")
    print(f"Val:   {len(val)} examples -> {val_path}")

    # Print stats
    sources = {}
    difficulties = {}
    for d in dataset:
        sources[d["source"]] = sources.get(d["source"], 0) + 1
        difficulties[d["difficulty"]] = difficulties.get(d["difficulty"], 0) + 1

    print("\nBy source:", sources)
    print("By difficulty:", difficulties)


if __name__ == "__main__":
    main()
