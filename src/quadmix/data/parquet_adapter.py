"""
Parquet data adapter — reads/writes parquet format datasets.
"""

from typing import List, Optional, Dict, Any
import os

import numpy as np
import pandas as pd

from quadmix.data.base import BaseDataAdapter, UnifiedData


class ParquetDataAdapter(BaseDataAdapter):
    """
    Adapter for Parquet format datasets.
    
    Auto-detects column naming:
      - 'text' or 'content' or 'document' for document text
      - 'domain' or 'label' or 'category' for domain labels
      - 'token_count' or 'length' for token counts
    
    Usage:
        adapter = ParquetDataAdapter()
        data = adapter.load("data/shard_00000.parquet")
    """

    def can_handle(self, path: str) -> bool:
        return path.endswith(".parquet")

    def load(
        self,
        path: str,
        text_column: Optional[str] = None,
        domain_column: Optional[str] = None,
        token_count_column: Optional[str] = None,
        max_docs: Optional[int] = None,
        **kwargs,
    ) -> UnifiedData:
        """
        Load a parquet file.
        
        Args:
            path: Path to .parquet file.
            text_column: Column name for text. Auto-detected if None.
            domain_column: Column name for domain labels. Skipped if None.
            token_count_column: Column name for precomputed token counts.
            max_docs: Limit number of documents loaded.
            
        Returns:
            UnifiedData instance.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Parquet file not found: {path}")

        df = pd.read_parquet(path, **kwargs)
        if max_docs is not None and max_docs < len(df):
            df = df.head(max_docs).reset_index(drop=True)

        # Auto-detect text column
        text_col = text_column or self._detect_text_column(df)
        if text_col is None:
            raise ValueError(
                f"Cannot detect text column in {list(df.columns)}. "
                f"Please specify text_column= explicitly."
            )

        texts = df[text_col].astype(str).tolist()

        # Auto-detect domain column
        domain_labels = None
        if domain_column:
            col = df[domain_column]
            if col.dtype.kind in ('i', 'u'):
                domain_labels = col.to_numpy(dtype=np.int64)
            elif col.dtype.kind in ('O', 'U', 'S'):
                domain_labels = col.astype('category').cat.codes.to_numpy(dtype=np.int64)
            else:
                raise TypeError(f"Domain column '{domain_column}' has unsupported dtype {col.dtype}")
        else:
            for candidate in ["domain", "label", "category", "domain_id"]:
                if candidate in df.columns:
                    col = df[candidate]
                    if col.dtype.kind in ('i', 'u'):
                        domain_labels = col.to_numpy(dtype=np.int64)
                    elif col.dtype.kind in ('O', 'U', 'S'):
                        domain_labels = col.astype('category').cat.codes.to_numpy(dtype=np.int64)
                    if domain_labels is not None:
                        break

        # Token counts
        token_counts = None
        if token_count_column and token_count_column in df.columns:
            token_counts = df[token_count_column].to_numpy(dtype=np.int64)
        elif "token_count" in df.columns:
            token_counts = df["token_count"].to_numpy(dtype=np.int64)
        elif "length" in df.columns:
            token_counts = df["length"].to_numpy(dtype=np.int64)

        # Estimate if not available
        if token_counts is None:
            token_counts = np.array(
                [self._estimate_token_count(t) for t in texts],
                dtype=np.int64,
            )

        doc_ids = self._generate_doc_ids(texts)

        return UnifiedData(
            texts=texts,
            doc_ids=doc_ids,
            token_counts=token_counts,
            domain_labels=domain_labels,
            metadata={
                "source": path,
                "format": "parquet",
                "num_docs": len(texts),
                "columns": list(df.columns),
                "text_column": text_col,
            },
        )

    def save(
        self,
        data: UnifiedData,
        path: str,
        include_text: bool = True,
        include_domain: bool = True,
        include_quality: bool = False,
        **kwargs,
    ):
        """Save UnifiedData to parquet."""
        records = {}
        if include_text:
            records["text"] = data.texts
        records["doc_id"] = data.doc_ids
        if data.token_counts is not None and include_text:
            records["token_count"] = data.token_counts.tolist()
        if data.domain_labels is not None and include_domain:
            records["domain"] = data.domain_labels.tolist()
        if data.quality_scores is not None and include_quality:
            for n in range(data.quality_scores.shape[1]):
                records[f"quality_score_{n}"] = data.quality_scores[:, n].tolist()

        df = pd.DataFrame(records)
        df.to_parquet(path, index=False, **kwargs)

    @staticmethod
    def _detect_text_column(df: pd.DataFrame) -> Optional[str]:
        """Detect the text column from common naming conventions."""
        candidates = ["text", "content", "document", "doc", "body", "sentence"]
        for col in candidates:
            if col in df.columns:
                return col
        # Fallback: first string column
        for col in df.columns:
            if df[col].dtype == "object" or isinstance(df[col].iloc[0], str):
                return col
        return None
