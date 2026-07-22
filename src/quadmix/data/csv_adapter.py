"""
CSV data adapter — reads/writes CSV format datasets.
"""

from typing import Optional
import os

import numpy as np
import pandas as pd

from quadmix.data.base import BaseDataAdapter, UnifiedData


class CSVDataAdapter(BaseDataAdapter):
    """Adapter for CSV format datasets."""

    def can_handle(self, path: str) -> bool:
        return path.endswith(".csv")

    def load(
        self,
        path: str,
        text_column: Optional[str] = None,
        domain_column: Optional[str] = None,
        max_docs: Optional[int] = None,
        encoding: str = "utf-8",
        **kwargs,
    ) -> UnifiedData:
        """Load a CSV file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")

        df = pd.read_csv(path, encoding=encoding, **kwargs)
        if max_docs is not None and max_docs < len(df):
            df = df.head(max_docs).reset_index(drop=True)

        # Detect text column
        text_col = text_column or self._detect_text_column(df)
        if text_col is None:
            raise ValueError(
                f"Cannot detect text column in {list(df.columns)}. "
                f"Specify text_column= explicitly."
            )

        texts = df[text_col].astype(str).tolist()
        doc_ids = self._generate_doc_ids(texts)
        token_counts = np.array(
            [self._estimate_token_count(t) for t in texts],
            dtype=np.int64,
        )

        # Domain column
        domain_labels = None
        if domain_column:
            col = df[domain_column]
            if col.dtype.kind in ('i', 'u'):
                domain_labels = col.to_numpy(dtype=np.int64)
            elif col.dtype.kind in ('O', 'U', 'S'):
                domain_labels = col.astype('category').cat.codes.to_numpy(dtype=np.int64)
            else:
                raise TypeError(f"Domain column '{domain_column}' has unsupported dtype {col.dtype}")
        elif "domain" in df.columns:
            col = df["domain"]
            if col.dtype.kind in ('i', 'u'):
                domain_labels = col.to_numpy(dtype=np.int64)
            elif col.dtype.kind in ('O', 'U', 'S'):
                domain_labels = col.astype('category').cat.codes.to_numpy(dtype=np.int64)

        return UnifiedData(
            texts=texts,
            doc_ids=doc_ids,
            token_counts=token_counts,
            domain_labels=domain_labels,
            metadata={
                "source": path,
                "format": "csv",
                "num_docs": len(texts),
                "columns": list(df.columns),
                "text_column": text_col,
            },
        )

    def save(self, data: UnifiedData, path: str, **kwargs):
        """Save UnifiedData to CSV."""
        df = data.to_dataframe()
        df.to_csv(path, index=False, **kwargs)

    @staticmethod
    def _detect_text_column(df: pd.DataFrame) -> Optional[str]:
        """Auto-detect text column."""
        candidates = ["text", "content", "document", "doc", "body", "sentence"]
        for col in candidates:
            if col in df.columns:
                return col
        for col in df.columns:
            if df[col].dtype == "object":
                return col
        return None
