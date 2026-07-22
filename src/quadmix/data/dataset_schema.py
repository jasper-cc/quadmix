"""
DatasetSchema — YAML-driven configuration that maps parquet column names
to algorithm concepts (domain, quality, text, char_count).

Decouples QuaDMix from any specific dataset schema so that arbitrary
datasets can be run with zero code changes — just write a YAML config.

Usage:
    # From YAML file (唯一入口)
    schema = DatasetSchema.from_yaml("configs/schema_essential_web.yaml")

    # Validate against a parquet file
    schema._validate(parquet_columns, parquet_dtypes)

    # Access parsed config
    schema.domain_col        # "domain" (from YAML)
    schema.quality_cols      # ["qs_dclm", ...] (from YAML)
    schema.quality_directions # [True, True, ..., False]  (higher_better)
    schema.text_col          # "text" (from YAML)
    schema.char_count_col    # "doc_char_count" or None (from YAML)

DatasetSchema() 无参构造会报错 — 必须通过 from_yaml() 加载配置。
示例配置: configs/schema_essential_web.yaml
"""

import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


def _parse_quality_cols(raw: list) -> Tuple[List[str], List[bool]]:
    """Parse quality_cols from YAML, supporting both plain strings and
    {name: ..., higher_better: ...} dicts.

    Returns (col_names, higher_better_flags).
    """
    names: List[str] = []
    directions: List[bool] = []
    for entry in raw:
        if isinstance(entry, str):
            names.append(entry)
            directions.append(True)
        elif isinstance(entry, dict):
            names.append(entry["name"])
            directions.append(entry.get("higher_better", True))
        else:
            raise ValueError(f"quality_cols entry must be str or dict, got {type(entry)}: {entry}")
    return names, directions


@dataclass
class DatasetSchema:
    domain_col: Optional[str] = None
    quality_cols: Optional[List[str]] = None
    quality_directions: Optional[List[bool]] = None
    text_col: Optional[str] = None
    char_count_col: Optional[str] = None
    row_in_shard_col: Optional[str] = None
    domain_names: Optional[List[str]] = None
    quality_names: Optional[List[str]] = None

    def __post_init__(self):
        if self.domain_col is None and self.quality_cols is None and self.text_col is None:
            raise ValueError(
                "DatasetSchema() 无参构造不可用。请通过 from_yaml() 加载配置文件。\n"
                "示例: schema = DatasetSchema.from_yaml('configs/schema_essential_web.yaml')\n"
                "详见 configs/ 目录下的示例 YAML 配置。"
            )
        if self.domain_col is None:
            raise ValueError("DatasetSchema.domain_col 是必填项。")
        if self.text_col is None:
            raise ValueError("DatasetSchema.text_col 是必填项。")
        if self.quality_cols is None or len(self.quality_cols) == 0:
            raise ValueError("DatasetSchema.quality_cols 是必填项且不可为空列表。")

    @classmethod
    def from_yaml(cls, path: str) -> "DatasetSchema":
        """Load DatasetSchema from a YAML file."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ValueError(
                f"YAML 配置文件为空: {path}\n"
                "请填写 domain_col, quality_cols, text_col 等字段。\n"
                "示例: configs/schema_essential_web.yaml"
            )

        raw_quality = data.get("quality_cols")
        if raw_quality is None:
            raise ValueError(
                f"YAML 配置缺少 quality_cols 字段: {path}\n"
                "quality_cols 是必填项，指定数据集的质量评分列名。"
            )

        col_names, directions = _parse_quality_cols(raw_quality)

        return cls(
            domain_col=data.get("domain_col"),
            quality_cols=col_names,
            quality_directions=directions,
            text_col=data.get("text_col"),
            char_count_col=data.get("char_count_col"),
            row_in_shard_col=data.get("row_in_shard_col"),
            domain_names=data.get("domain_names"),
            quality_names=data.get("quality_names"),
        )

    def _validate(
        self,
        parquet_columns: List[str],
        parquet_dtypes: Dict[str, str],
    ) -> None:
        """Check that all specified columns exist in the parquet with correct dtype.

        Raises ValueError with available columns if any mismatch.
        """
        missing = []
        dtype_mismatch = []

        required_cols = [self.domain_col] + self.quality_cols
        if self.text_col:
            required_cols.append(self.text_col)
        if self.char_count_col is not None:
            required_cols.append(self.char_count_col)

        for col in required_cols:
            if col not in parquet_columns:
                missing.append(col)
            elif col == self.domain_col:
                pass
            elif col in self.quality_cols:
                dt = parquet_dtypes.get(col, "")
                if not dt.startswith(("float", "double", "Float64", "Float32")):
                    dtype_mismatch.append((col, dt, "float"))
            elif col == self.char_count_col:
                dt = parquet_dtypes.get(col, "")
                if not dt.startswith(("int", "Int64", "Int32", "uint", "UInt64")):
                    dtype_mismatch.append((col, dt, "int"))

        row_warnings = []
        if self.row_in_shard_col is not None and self.row_in_shard_col not in parquet_columns:
            row_warnings.append(self.row_in_shard_col)

        if missing or dtype_mismatch:
            lines = ["Schema 校验失败:"]
            if missing:
                lines.append(f"  以下列不存在于 parquet 中:")
                for col in missing:
                    lines.append(f"    - {col} (缺失)")
            if dtype_mismatch:
                lines.append(f"  以下列 dtype 不匹配:")
                for col, actual, expected in dtype_mismatch:
                    lines.append(f"    - {col}: 实际={actual}, 期望={expected}")

            lines.append("")
            lines.append(f"可用列 ({len(parquet_columns)}):")
            for col in parquet_columns:
                dt = parquet_dtypes.get(col, "?")
                lines.append(f"  {col} ({dt})")
            lines.append("")
            lines.append("请创建 YAML 配置文件并使用 --schema 指定。")
            lines.append("示例:")
            lines.append(f"  domain_col: <one_of_available>")
            lines.append(f"  quality_cols: [<float_cols>]")
            lines.append(f"  text_col: <text_col_if_available>")
            raise ValueError("\n".join(lines))

        if row_warnings:
            import warnings
            for col in row_warnings:
                warnings.warn(
                    f"row_in_shard_col '{col}' 不存在于 parquet 中。"
                    f"将使用位置索引读取文本（较慢）。"
                    f"如无该列，请在 YAML 中设置 row_in_shard_col: null。",
                    UserWarning,
                )

    def needs_text_for_char_count(self) -> bool:
        """True when char_count_col is None and must be computed from text."""
        return self.char_count_col is None

    def needs_text_loading(self) -> bool:
        """True when text column must be loaded during metadata reading
        (either for char_count computation or because it's explicitly needed).
        """
        return self.needs_text_for_char_count()

    def metadata_read_columns(self) -> List[str]:
        """Columns to read from parquet for metadata loading."""
        cols = [self.domain_col] + list(self.quality_cols)
        if self.char_count_col is not None:
            cols.append(self.char_count_col)
        if self.needs_text_for_char_count():
            cols.append(self.text_col)
        if self.row_in_shard_col is not None:
            cols.append(self.row_in_shard_col)
        return cols

    def text_read_columns(self) -> List[str]:
        """Columns to read from parquet for text/tokenize loading."""
        cols = [self.text_col]
        if self.row_in_shard_col is not None:
            cols.append(self.row_in_shard_col)
        return cols
