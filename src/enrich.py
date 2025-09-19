import pandas as pd

# 1.1. Join transaction với product attributes
def enrich_product_transaction(df: pd.DataFrame, product_dim: pd.DataFrame) -> pd.DataFrame:

    return df.merge(product_dim, on="product_id", how="left")

# 1.1. Join transaction với customer data (optional).
def enrich_product_customer(df: pd.DataFrame, customer_dim: pd.DataFrame) -> pd.DataFrame:

    return df.merge(customer_dim, on="customer_id", how="left")

# 1.3. Join transaction với promotion data (nếu muốn phân tích uplift)
def enrich_product_promotion(df: pd.DataFrame, promo_dim: pd.DataFrame) -> pd.DataFrame:

    return df.merge(promo_dim, on="promotion_id", how="left")
