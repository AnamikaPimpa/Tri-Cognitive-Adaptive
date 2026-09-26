"""Validate timestamp integrity and missing values in the Dok Khamtai CSV.

This script performs checks only. It does not fill, interpolate, delete, sort,
recalculate or write any data.

Checks:
* duplicate timestamps;
* missing timestamps in the expected interval; and
* missing values in every column.

Exit status is 0 when all checks pass and 1 when any issue is found, making the
script suitable for use as a validation step in a data pipeline.

Usage:
    python services/preprocess_dok_khamtai_data.py
    python services/preprocess_dok_khamtai_data.py --frequency 15min
    python services/preprocess_dok_khamtai_data.py --input helpers/weather.csv
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = (
    PROJECT_ROOT / "helpers" / "dok_khamtai_15min_2020-07-01_2025-06-30.csv"
)
DEFAULT_TIMESTAMP_COLUMN = "timestamp"
DEFAULT_FREQUENCY = "15min"
TIMEZONE = "Asia/Bangkok"
MAX_EXAMPLES = 5


@dataclass
class ValidationReport:
    """Results from the three requested data-quality checks."""

    path: Path
    row_count: int
    column_count: int
    timestamp_column: str
    frequency: str
    first_timestamp: Optional[pd.Timestamp]
    last_timestamp: Optional[pd.Timestamp]
    duplicate_rows: int
    duplicate_timestamps: int
    duplicate_examples: List[str]
    missing_timestamp_count: int
    missing_timestamp_examples: List[str]
    missing_values: Dict[str, int]
    invalid_timestamp_count: int

    @property
    def total_missing_cells(self) -> int:
        return sum(self.missing_values.values())

    @property
    def passed(self) -> bool:
        return not any(
            (
                self.duplicate_rows,
                self.missing_timestamp_count,
                self.total_missing_cells,
                self.invalid_timestamp_count,
            )
        )


def normalize_blank_strings(data: pd.DataFrame) -> pd.DataFrame:
    """Treat empty and whitespace-only text cells as missing values."""
    result = data.copy()
    text_columns = result.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        blank = result[column].notna() & result[column].astype(str).str.strip().eq("")
        result.loc[blank, column] = pd.NA
    return result


def validate_csv(
    path: Path,
    timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN,
    frequency: str = DEFAULT_FREQUENCY,
) -> ValidationReport:
    """Read a CSV and run duplicate, interval-gap and missing-value checks."""
    if not path.exists():
        raise FileNotFoundError(f"input file does not exist: {path}")

    try:
        interval = pd.Timedelta(frequency)
    except ValueError as exc:
        raise ValueError(f"invalid frequency: {frequency}") from exc
    if interval <= pd.Timedelta(0):
        raise ValueError("frequency must be greater than zero")

    data = normalize_blank_strings(pd.read_csv(path))
    if timestamp_column not in data.columns:
        raise ValueError(
            f"timestamp column {timestamp_column!r} is absent; "
            f"available columns: {list(data.columns)}"
        )

    missing_values = data.isna().sum().astype(int).to_dict()

    timestamp_missing = data[timestamp_column].isna()
    parsed = pd.to_datetime(
        data[timestamp_column],
        errors="coerce",
        utc=True,
    ).dt.tz_convert(TIMEZONE)
    invalid_timestamp = parsed.isna() & ~timestamp_missing
    valid_timestamps = parsed.dropna()

    duplicate_mask = valid_timestamps.duplicated(keep=False)
    duplicate_values = valid_timestamps[duplicate_mask]
    duplicate_examples = [
        timestamp.isoformat()
        for timestamp in duplicate_values.drop_duplicates().head(MAX_EXAMPLES)
    ]

    unique_timestamps = pd.DatetimeIndex(valid_timestamps.drop_duplicates()).sort_values()
    first_timestamp = unique_timestamps[0] if len(unique_timestamps) else None
    last_timestamp = unique_timestamps[-1] if len(unique_timestamps) else None

    if len(unique_timestamps) >= 2:
        expected = pd.date_range(
            first_timestamp,
            last_timestamp,
            freq=interval,
        )
        missing_timestamps = expected.difference(unique_timestamps)
    else:
        missing_timestamps = pd.DatetimeIndex([])

    return ValidationReport(
        path=path,
        row_count=len(data),
        column_count=len(data.columns),
        timestamp_column=timestamp_column,
        frequency=frequency,
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        duplicate_rows=int(duplicate_mask.sum()),
        duplicate_timestamps=int(duplicate_values.nunique()),
        duplicate_examples=duplicate_examples,
        missing_timestamp_count=len(missing_timestamps),
        missing_timestamp_examples=[
            timestamp.isoformat() for timestamp in missing_timestamps[:MAX_EXAMPLES]
        ],
        missing_values=missing_values,
        invalid_timestamp_count=int(invalid_timestamp.sum()),
    )


def status_label(issue_count: int) -> str:
    return "PASS" if issue_count == 0 else "FAIL"


def print_report(report: ValidationReport) -> None:
    """Print a concise, human-readable validation report."""
    print(f"file: {report.path}")
    print(f"rows: {report.row_count:,}   columns: {report.column_count:,}")
    if report.first_timestamp is not None:
        print(f"range: {report.first_timestamp} -> {report.last_timestamp}")
    else:
        print("range: no valid timestamps")
    print(f"expected frequency: {report.frequency}\n")

    print(
        f"[{status_label(report.duplicate_rows)}] duplicate timestamps: "
        f"{report.duplicate_timestamps:,} values, "
        f"{report.duplicate_rows:,} affected rows"
    )
    for example in report.duplicate_examples:
        print(f"  {example}")

    print(
        f"[{status_label(report.missing_timestamp_count)}] missing timestamps: "
        f"{report.missing_timestamp_count:,}"
    )
    for example in report.missing_timestamp_examples:
        print(f"  {example}")

    print(
        f"[{status_label(report.total_missing_cells)}] missing values: "
        f"{report.total_missing_cells:,} cells"
    )
    for column, count in report.missing_values.items():
        if count:
            print(f"  {column}: {count:,}")

    if report.invalid_timestamp_count:
        print(
            f"[FAIL] invalid timestamp text: "
            f"{report.invalid_timestamp_count:,} rows"
        )

    print(f"\noverall: {'PASS' if report.passed else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--timestamp-column", default=DEFAULT_TIMESTAMP_COLUMN)
    parser.add_argument("--frequency", default=DEFAULT_FREQUENCY)
    args = parser.parse_args()

    try:
        report = validate_csv(args.input, args.timestamp_column, args.frequency)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        raise SystemExit(f"validation could not run: {exc}") from exc

    print_report(report)
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
