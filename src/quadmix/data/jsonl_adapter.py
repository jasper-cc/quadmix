"""
JSONL data adapter — reads/writes JSONL format datasets.
"""

from typing import Optional
import json
import os

import numpy as np
import pandas as pd

from quadmix.data.base import BaseDataAdapter, UnifiedData


class JSONLDataAdapter(BaseDataAdapter):
    """Adapter for JSONL (JSON Lines) datasets."""

    def can_handle(self, path: str) -> bool:
        return path.endswith(".jsonl") or path.endswith(".jsonlines") or path.endswith(".ndjson")

    def load(
        self,
        path: str,
        text_key: Optional[str] = None,
        domain_key: Optional[str] = None,
        max_docs: Optional[int] = None,
        **kwargs,
    ) -> UnifiedData:
        """
        Load a JSONL file.
        
        Args:
            path: Path to .jsonl file.
            text_key: Key for text field. Auto-detected if None.
            domain_key: Key for domain label. Skipped if None.
            max_docs: Limit number of documents.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"JSONL file not found: {path}")

        texts = []
        domain_labels_list = []
        all_keys = set()

        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_docs is not None and i >= max_docs:
                    break
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                all_keys.update(record.keys())

                if text_key:
                    text = record.get(text_key, "")
                else:
                    detected_key = self._detect_text_key_name(record)
                    text = str(record.get(detected_key, "")) if detected_key else self._extract_text_fallback(record)
                texts.append(text)

                if domain_key:
                    domain_labels_list.append(record.get(domain_key, -1))

        if not text_key:
            text_key = self._detect_text_key_name({k: "" for k in all_keys})

        doc_ids = self._generate_doc_ids(texts)
        token_counts = np.array(
            [self._estimate_token_count(t) for t in texts],
            dtype=np.int64,
        )

        domain_labels = None
        if domain_labels_list:
            domain_labels = np.array(domain_labels_list, dtype=np.int64)

        return UnifiedData(
            texts=texts,
            doc_ids=doc_ids,
            token_counts=token_counts,
            domain_labels=domain_labels,
            metadata={
                "source": path,
                "format": "jsonl",
                "num_docs": len(texts),
                "detected_text_key": text_key,
                "all_keys": list(all_keys),
            },
        )

    def save(
        self,
        data: UnifiedData,
        path: str,
        extra_fields: Optional[dict] = None,
        **kwargs,
    ):
        """Save UnifiedData to JSONL."""
        with open(path, "w", encoding="utf-8") as f:
            for i in range(len(data)):
                record = {
                    "text": data.texts[i],
                    "doc_id": data.doc_ids[i],
                }
                if data.token_counts is not None:
                    record["token_count"] = int(data.token_counts[i])
                if data.domain_labels is not None:
                    record["domain"] = int(data.domain_labels[i])
                if extra_fields:
                    record.update(extra_fields)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _detect_text_key_name(record: dict) -> str:
        """Auto-detect the text KEY NAME from common conventions."""
        for key in ["text", "content", "document", "doc", "body", "sentence", "input"]:
            if key in record:
                return key
        for k, v in record.items():
            if isinstance(v, str) and len(v) > 20:
                return k
        return ""

    @staticmethod
    def _extract_text_fallback(record: dict) -> str:
        """Extract text value when no standard key is found."""
        for key in ["text", "content", "document", "doc", "body", "sentence", "input"]:
            if key in record:
                return str(record[key])
        for v in record.values():
            if isinstance(v, str) and len(v) > 20:
                return v
        return ""
