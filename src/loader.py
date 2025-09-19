# src/loader/loader.py

import os
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

# Nếu dùng BigQuery
try:
    from google.cloud import bigquery
except ImportError:
    bigquery = None


class BaseLoader(ABC):
    """Abstract Loader: define chuẩn cho các loader."""

    @abstractmethod
    def load(self, **kwargs) -> pd.DataFrame:
        pass


# 2. Loader từ CSV
class CSVLoader(BaseLoader):

    # 2.1. Initialize CSVLoader
    def __init__(self, filepath: str, sep: str = ","):
        self.filepath = filepath
        self.sep = sep

    def load(self, **kwargs) -> pd.DataFrame:
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"CSV file not found: {self.filepath}")
        df = pd.read_csv(self.filepath, sep=self.sep)
        return df

# 3. BigQuery Loader
class BigQueryLoader(BaseLoader):

# 3.1. Initialize Google BigQuery Sessions
    def __init__(self, project_id: str, dataset_id: str, table_id: Optional[str] = None):
        if bigquery is None:
            raise ImportError("[LOADER] Google BigQuey Client is not initialized.")
        self.client = bigquery.Client(project=project_id)
        self.dataset_id = dataset_id
        self.table_id = table_id

    def load(self, query: Optional[str] = None, **kwargs) -> pd.DataFrame:
        if query is None:
            if not self.table_id:
                raise ValueError("Cần truyền table_id nếu không có query.")
            full_table = f"{self.client.project}.{self.dataset_id}.{self.table_id}"
            query = f"SELECT * FROM `{full_table}`"
        df = self.client.query(query).to_dataframe()
        return df


def get_loader(config: Dict[str, Any]) -> BaseLoader:
    """Factory function để chọn loader từ config."""

    loader_type = config.get("type")
    if loader_type == "csv":
        return CSVLoader(
            filepath=config["filepath"],
            sep=config.get("sep", ",")
        )
    elif loader_type == "bigquery":
        return BigQueryLoader(
            project_id=config["project_id"],
            dataset_id=config["dataset_id"],
            table_id=config.get("table_id")
        )
    else:
        raise ValueError(f"Loader type '{loader_type}' chưa hỗ trợ.")


if __name__ == "__main__":
    # Example: test CSVLoader
    config_csv = {
        "type": "csv",
        "filepath": "data/sku_sample.csv"
    }
    loader = get_loader(config_csv)
    df = loader.load()
    print(df.head())
